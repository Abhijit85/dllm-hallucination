"""
LLaDA parallel denoising harness.

Wraps LLaDA-8B (masked diffusion LM) to:
  1. Run N independent denoising chains from the same masked input
  2. Support pluggable demasking-order strategies (learned | random | hybrid)
  3. Return lightweight per-step summaries for analysis without storing logits

Model: server-local LLaDA checkpoints only
Paper: https://arxiv.org/abs/2406.11838
"""

from __future__ import annotations

import importlib
import json
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    import transformers.modeling_rope_utils as rope_utils
    from transformers import AutoModel, AutoTokenizer
    from transformers.modeling_utils import PreTrainedModel
except ImportError:  # pragma: no cover - exercised only when optional deps are missing
    AutoModel = None
    AutoTokenizer = None
    PreTrainedModel = None
    rope_utils = None


# ── constants ──────────────────────────────────────────────────────────────────
DEFAULT_MASK_TOKEN_ID = 126336  # LLaDA's [MASK] id
MASK_TOKEN_ID = DEFAULT_MASK_TOKEN_ID
MODEL_ID = "GSAI-ML/LLaDA-8B-Instruct"

# Short aliases → filesystem paths used during the original paper experiments.
# These paths are lab-specific and will not exist on your machine.
#
# For external use, override in one of two ways:
#   1. Environment variable:  export LLADA_MODEL_PATH=/your/local/path/to/LLaDA-8B-Instruct
#   2. CLI argument:          python run_experiment.py --model_id /your/local/path/...
#
# Models can be downloaded from Hugging Face:
#   LLaDA-8B-Instruct: https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct
#   Dream-v0-Instruct: https://huggingface.co/Dream-org/Dream-v0-Instruct-7B
LOCAL_LLADA_MODEL_DIRS = {
    "llada-8b-instruct": "/mnt/shared/shared_hf_home/hub/GSAI-ML--LLaDA-8B-Instruct",
    "GSAI-ML/LLaDA-8B-Instruct": "/mnt/shared/shared_hf_home/hub/GSAI-ML--LLaDA-8B-Instruct",
    "llada-8b-base": "/mnt/shared/shared_hf_home/hub/GSAI-ML--LLaDA-8B-Base",
    "GSAI-ML/LLaDA-8B-Base": "/mnt/shared/shared_hf_home/hub/GSAI-ML--LLaDA-8B-Base",
    "Dream-org/Dream-v0-Instruct-7B": "/mnt/shared/shared_hf_home/hub/models--Dream-org--Dream-v0-Instruct-7B",
    "dream-7b-instruct": "/mnt/shared/shared_hf_home/hub/models--Dream-org--Dream-v0-Instruct-7B",
    "Dream-org/Dream-v0-Base-7B": "/mnt/shared/shared_hf_home/hub/models--Dream-org--Dream-v0-Base-7B",
    "dream-7b-base": "/mnt/shared/shared_hf_home/hub/models--Dream-org--Dream-v0-Base-7B",
}
DEFAULT_LOCAL_MODEL_ALIAS = "llada-8b-instruct"


def available_local_llada_models() -> dict[str, str]:
    """Return the configured server-local LLaDA directories."""
    return dict(LOCAL_LLADA_MODEL_DIRS)


def _resolve_hf_snapshot_dir(model_dir: Path) -> Path:
    """
    Resolve a Hugging Face cache model directory to a concrete snapshot dir.

    Supports standard cache layout:
      model_dir/
        refs/main
        snapshots/<sha>/
    """
    if (model_dir / "config.json").exists():
        return model_dir

    refs_main = model_dir / "refs" / "main"
    if refs_main.exists():
        snapshot_name = refs_main.read_text().strip()
        snapshot_dir = model_dir / "snapshots" / snapshot_name
        if snapshot_dir.exists():
            return snapshot_dir

    snapshots_dir = model_dir / "snapshots"
    if snapshots_dir.exists():
        snapshot_dirs = sorted(p for p in snapshots_dir.iterdir() if p.is_dir())
        if snapshot_dirs:
            return snapshot_dirs[-1]

    return model_dir


def resolve_mask_token_id(model_path: str) -> int:
    """
    Resolve the active mask token id for the loaded diffusion model.

    Priority:
      1. DLLM_MASK_TOKEN_ID env var
      2. model config.json mask_token_id
      3. default LLaDA mask token id
    """
    env_override = os.environ.get("DLLM_MASK_TOKEN_ID")
    if env_override is not None:
        return int(env_override)

    cfg_path = Path(model_path) / "config.json"
    if cfg_path.exists():
        try:
            with open(cfg_path) as f:
                cfg = json.load(f)
            mask_token_id = cfg.get("mask_token_id")
            if mask_token_id is not None:
                return int(mask_token_id)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass

    return DEFAULT_MASK_TOKEN_ID


def resolve_local_llada_model(model_ref: str | None = None) -> str:
    """
    Resolve a model reference to a server-local filesystem path.

    Accepted values:
      - known aliases/HF ids listed in LOCAL_LLADA_MODEL_DIRS
      - explicit absolute local filesystem paths
      - environment variable LLADA_MODEL_PATH / LLADA_MODEL_DIR
    """
    env_override = os.environ.get("LLADA_MODEL_PATH") or os.environ.get("LLADA_MODEL_DIR")
    requested = model_ref or env_override or DEFAULT_LOCAL_MODEL_ALIAS

    if requested in LOCAL_LLADA_MODEL_DIRS:
        resolved = Path(LOCAL_LLADA_MODEL_DIRS[requested])
    else:
        candidate = Path(requested).expanduser()
        if not candidate.is_absolute():
            known = ", ".join(sorted(LOCAL_LLADA_MODEL_DIRS))
            raise ValueError(
                "LLaDA models must be loaded from server-local paths only. "
                f"Unknown model reference '{requested}'. "
                f"Use one of: {known}, or pass an absolute local path."
            )
        resolved = candidate

    resolved = _resolve_hf_snapshot_dir(resolved)

    if not resolved.exists():
        raise FileNotFoundError(
            f"Resolved LLaDA path does not exist on this server: {resolved}"
        )
    if not resolved.is_dir():
        raise ValueError(f"LLaDA path must be a directory: {resolved}")

    required_files = ("config.json",)
    missing = [name for name in required_files if not (resolved / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"LLaDA directory {resolved} is missing required files: {', '.join(missing)}"
        )

    return str(resolved)


def _patch_llada_transformers_compat() -> None:
    """
    Patch transformers loading for local LLaDA remote-code classes.

    Some local LLaDA checkpoints ship a model class that defines `tie_weights()`
    but not the newer `all_tied_weights_keys` attribute expected by recent
    transformers during `from_pretrained()`.
    """
    original = getattr(PreTrainedModel, "mark_tied_weights_as_initialized", None)
    if original is None:
        return
    if getattr(original, "_dllm_llada_compat_patched", False):
        return

    def patched_mark_tied_weights_as_initialized(self, *args, **kwargs):
        if not hasattr(self, "all_tied_weights_keys"):
            tied_keys = getattr(self, "_tied_weights_keys", None) or []
            if isinstance(tied_keys, (list, tuple, set)):
                self.all_tied_weights_keys = {str(key): True for key in tied_keys}
            else:
                self.all_tied_weights_keys = {}
        return original(self, *args, **kwargs)

    patched_mark_tied_weights_as_initialized._dllm_llada_compat_patched = True
    PreTrainedModel.mark_tied_weights_as_initialized = patched_mark_tied_weights_as_initialized

    original_finalize = getattr(PreTrainedModel, "_finalize_model_loading", None)
    if original_finalize is not None and not getattr(original_finalize, "_dllm_llada_compat_patched", False):
        def patched_finalize_model_loading(cls, model, load_config, loading_info):
            try:
                return original_finalize(model, load_config, loading_info)
            except TypeError as exc:
                if "unexpected keyword argument 'missing_keys'" not in str(exc):
                    raise
                model.tie_weights()
                return loading_info

        patched_finalize_model_loading._dllm_llada_compat_patched = True
        PreTrainedModel._finalize_model_loading = classmethod(patched_finalize_model_loading)

    # transformers 5.3.0 advertises rope_type="default" in docs/validation,
    # but Dream remote code still expects it to be present in ROPE_INIT_FUNCTIONS.
    # Add the missing backward-compat entry before loading Dream.
    if "default" not in rope_utils.ROPE_INIT_FUNCTIONS:
        def _compute_default_rope_parameters(config, device, seq_len=None, **rope_kwargs):
            if rope_kwargs:
                base = rope_kwargs["base"]
                dim = rope_kwargs["dim"]
            else:
                config.standardize_rope_params()
                rope_parameters = config.rope_parameters
                base = rope_parameters["rope_theta"]
                partial_rotary_factor = rope_parameters.get("partial_rotary_factor", 1.0)
                head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
                dim = int(head_dim * partial_rotary_factor)

            inv_freq = 1.0 / (
                base ** (
                    torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim
                )
            )
            attention_factor = 1.0
            return inv_freq, attention_factor

        rope_utils.ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters


class DemaskingOrder(str, Enum):
    LEARNED     = "learned"     # highest model confidence first (default)
    RANDOM      = "random"      # uniform random order
    HYBRID      = "hybrid"      # learned for first 50% steps, random for last 50%
    ENTROPY     = "entropy"     # lowest entropy (most agreed-upon) first


# ── data structures ────────────────────────────────────────────────────────────

@dataclass
class DenoiseStep:
    """Lightweight per-step record to avoid storing full logits on GPU."""
    step: int
    mean_entropy: float
    n_still_masked: int
    unmasked_this_step: list[int]


@dataclass
class DenoisePath:
    """One complete denoising trajectory for one sample."""
    path_id: int
    order: DemaskingOrder
    final_tokens: torch.Tensor | None = None
    steps: list[DenoiseStep] = field(default_factory=list)

    @property
    def entropy_trajectory(self) -> list[float]:
        return [s.mean_entropy for s in self.steps]


@dataclass
class ParallelPathResult:
    """N paths for one (prompt, source_info) pair."""
    prompt_tokens: torch.Tensor          # (seq_len,)
    paths: list[DenoisePath]

    # ── aggregated signals ──

    def token_entropy(self) -> torch.Tensor:
        """
        Shannon entropy over N-path token distributions.
        Shape: (seq_len,)

        For each position we treat the N final token ids as a distribution
        (majority vote → probability), then compute H.

        High entropy  → paths disagree → potential hallucination.
        Low entropy   → paths agree    → confident token.
        """
        if not self.paths or self.paths[0].final_tokens is None:
            raise ValueError("ParallelPathResult has no completed denoising paths.")

        seq_len = self.paths[0].final_tokens.shape[0]
        device  = self.paths[0].final_tokens.device
        n       = len(self.paths)

        # (N, seq_len) final token ids
        stacked = torch.stack([p.final_tokens for p in self.paths], dim=0)

        # For each position, count how often each unique token appears
        entropy = torch.zeros(seq_len, device=device)
        for pos in range(seq_len):
            counts = torch.bincount(stacked[:, pos], minlength=1).float()
            probs  = counts[counts > 0] / n
            entropy[pos] = -(probs * probs.log()).sum()

        return entropy

    def majority_vote(self) -> torch.Tensor:
        """(seq_len,) most common token id across N paths."""
        stacked = torch.stack([p.final_tokens for p in self.paths], dim=0)
        return stacked.mode(dim=0).values

    def disagreement_mask(self, entropy_threshold: float = 0.5) -> torch.Tensor:
        """Boolean (seq_len,): True where entropy > threshold."""
        return self.token_entropy() > entropy_threshold


# ── model wrapper ──────────────────────────────────────────────────────────────

class LLaDAHarness:
    def __init__(
        self,
        model_id: str = MODEL_ID,
        device: str = "cuda",
        torch_dtype=torch.bfloat16,
    ):
        if AutoModel is None or AutoTokenizer is None or PreTrainedModel is None:
            raise ImportError(
                "The 'transformers' package is required to instantiate LLaDAHarness. "
                "Install project dependencies with `pip install -e .`."
            )
        self.model_ref = model_id
        self.model_path = resolve_local_llada_model(model_id)
        self.mask_token_id = resolve_mask_token_id(self.model_path)
        self.device = device
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        _patch_llada_transformers_compat()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=True,
        )
        self.model = AutoModel.from_pretrained(
            self.model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            local_files_only=True,
        ).to(device).eval()
        # Prefer the loaded config value when available. This covers cases where
        # pre-load resolution falls back to the default mask token id.
        config_mask_id = getattr(self.model.config, "mask_token_id", None)
        if config_mask_id is not None:
            self.mask_token_id = int(config_mask_id)
        self._normalize_model_config()

    def _normalize_model_config(self) -> None:
        """
        Fill in config fields expected by newer transformers/runtime code.
        """
        if not hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = False
        if not hasattr(self.model.config, "use_return_dict"):
            self.model.config.use_return_dict = True
        if self._is_dream():
            gen_cfg = getattr(self.model, "generation_config", None)
            if gen_cfg is not None:
                if getattr(gen_cfg, "mask_token_id", None) is None:
                    gen_cfg.mask_token_id = self.mask_token_id
                for name, default in [
                    ("eps", 1e-3),
                    ("steps", 512),
                    ("alg", "origin"),
                    ("alg_temp", None),
                    ("temperature", 0.0),
                    ("top_k", None),
                    ("top_p", None),
                    ("output_history", False),
                    ("return_dict_in_generate", False),
                    ("num_return_sequences", 1),
                ]:
                    if not hasattr(gen_cfg, name):
                        setattr(gen_cfg, name, default)

    def _build_dream_generation_config(
        self,
        max_length: int,
        num_steps: int,
        temperature: float,
    ):
        module_name = self.model.__class__.__module__.rsplit(".", 1)[0] + ".generation_utils"
        gen_module = importlib.import_module(module_name)
        DreamGenerationConfig = gen_module.DreamGenerationConfig
        return DreamGenerationConfig(
            max_length=max_length,
            steps=num_steps,
            temperature=temperature,
            alg="origin",
            eps=1e-3,
            output_history=False,
            mask_token_id=self.mask_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

    # ── core: single denoising step ────────────────────────────────────────────

    @torch.no_grad()
    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        One masked-LM forward pass.
        x:       (batch, seq_len) with the configured mask token id for masked positions
        returns: (batch, seq_len, vocab) logits
        """
        return self.model(input_ids=x).logits

    @staticmethod
    def _safe_softmax(logits: torch.Tensor) -> torch.Tensor:
        """
        Compute a numerically safe probability distribution from logits.
        Falls back to a one-hot argmax distribution if softmax becomes invalid.
        """
        safe_logits = torch.nan_to_num(logits.float(), nan=-1e4, posinf=1e4, neginf=-1e4)
        probs = F.softmax(safe_logits, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)

        denom = probs.sum(dim=-1, keepdim=True)
        invalid = denom.squeeze(-1) <= 0
        if invalid.ndim == 0:
            invalid = invalid.unsqueeze(0)
            probs = probs.unsqueeze(0)
            safe_logits = safe_logits.unsqueeze(0)
            squeeze_back = True
        else:
            squeeze_back = False

        if invalid.any():
            argmax_idx = safe_logits.argmax(dim=-1, keepdim=True)
            fallback = torch.zeros_like(probs)
            fallback.scatter_(-1, argmax_idx, 1.0)
            probs = torch.where(invalid.unsqueeze(-1), fallback, probs)

        probs = probs / probs.sum(dim=-1, keepdim=True)
        if squeeze_back:
            probs = probs.squeeze(0)
        return probs

    def _sample_token_id(
        self,
        logits: torch.Tensor,
        rng_sample: torch.Generator | None = None,
    ) -> torch.Tensor:
        masked_logits = logits.clone()
        special_ids = {self.mask_token_id}
        for special_id in [
            self.tokenizer.eos_token_id,
            self.tokenizer.pad_token_id,
            self.tokenizer.bos_token_id,
        ]:
            if special_id is not None:
                special_ids.add(int(special_id))
        for special_id in special_ids:
            if 0 <= special_id < masked_logits.shape[-1]:
                masked_logits[special_id] = -1e9

        probs = self._safe_softmax(masked_logits)
        # Keep token choice deterministic; path diversity should come from
        # reveal order rather than multinomial token noise.
        return probs.argmax(dim=-1, keepdim=True)

    # ── demasking-order strategies ─────────────────────────────────────────────

    def _pick_positions_to_unmask(
        self,
        x: torch.Tensor,             # (seq_len,)
        logits: torch.Tensor,        # (seq_len, vocab)
        n_to_unmask: int,
        order: DemaskingOrder,
        step: int,
        total_steps: int,
        rng_cpu: torch.Generator | None = None,
    ) -> torch.Tensor:
        """
        Returns a 1-D LongTensor of `n_to_unmask` positions to reveal.
        Only considers currently masked positions.
        """
        masked_positions = (x == self.mask_token_id).nonzero(as_tuple=True)[0]
        if len(masked_positions) == 0:
            return torch.tensor([], dtype=torch.long)
        n_to_unmask = min(n_to_unmask, len(masked_positions))

        if order == DemaskingOrder.RANDOM:
            perm = torch.randperm(len(masked_positions), generator=rng_cpu)
            return masked_positions[perm.to(masked_positions.device)[:n_to_unmask]]

        # Confidence scores: max softmax prob at each masked position
        probs = self._safe_softmax(logits[masked_positions])   # (M, vocab)
        confidence = probs.max(dim=-1).values                 # (M,)

        if order == DemaskingOrder.LEARNED:
            # High confidence → unmask first
            top_k = confidence.topk(n_to_unmask).indices
            return masked_positions[top_k]

        if order == DemaskingOrder.ENTROPY:
            # Low entropy → unmask first (most certain)
            ent = -(probs * (probs + 1e-10).log()).sum(-1)
            top_k = ent.topk(n_to_unmask, largest=False).indices
            return masked_positions[top_k]

        if order == DemaskingOrder.HYBRID:
            # First half of steps: learned; second half: random
            midpoint = total_steps // 2
            if step < midpoint:
                top_k = confidence.topk(n_to_unmask).indices
                return masked_positions[top_k]
            else:
                perm = torch.randperm(len(masked_positions), generator=rng_cpu)
                return masked_positions[perm.to(masked_positions.device)[:n_to_unmask]]

        raise ValueError(f"Unknown order: {order}")

    # ── single chain ───────────────────────────────────────────────────────────

    def _run_chain(
        self,
        prompt_ids: torch.Tensor,       # (seq_len,)
        gen_len: int,
        num_steps: int,
        order: DemaskingOrder,
        path_id: int,
        seed: int | None = None,
    ) -> DenoisePath:
        """
        Run one masked-diffusion denoising chain.

        The prompt tokens are frozen (never masked).
        The generation region starts fully masked.
        """
        rng_cpu = torch.Generator(device="cpu")
        rng_sample = torch.Generator(device=self.device)
        if seed is not None:
            rng_cpu.manual_seed(seed)
            rng_sample.manual_seed(seed)

        # Build initial x: [prompt | MASK * gen_len]
        gen_mask = torch.full((gen_len,), self.mask_token_id, dtype=torch.long, device=self.device)
        x = torch.cat([prompt_ids, gen_mask])                   # (seq_len,)

        prompt_len = len(prompt_ids)
        total_masked = gen_len
        # How many tokens to unmask per step (linearly spaced)
        unmask_schedule = self._build_schedule(total_masked, num_steps)

        path = DenoisePath(path_id=path_id, order=order)

        for step_idx, n_unmask in enumerate(unmask_schedule):
            logits = self._forward(x.unsqueeze(0)).squeeze(0)   # (seq_len, vocab)

            # Only operate on the generation region
            gen_x       = x[prompt_len:]
            gen_logits  = logits[prompt_len:]

            masked_in_gen = (gen_x == self.mask_token_id)
            if masked_in_gen.any():
                probs_masked = self._safe_softmax(gen_logits[masked_in_gen])
                step_entropy = float(
                    -(probs_masked * (probs_masked + 1e-10).log()).sum(-1).mean().item()
                )
            else:
                step_entropy = 0.0

            positions_to_unmask = self._pick_positions_to_unmask(
                x=gen_x,
                logits=gen_logits,
                n_to_unmask=n_unmask,
                order=order,
                step=step_idx,
                total_steps=num_steps,
                rng_cpu=rng_cpu,
            )

            # Sample token ids for the chosen positions
            for pos in positions_to_unmask:
                sampled = self._sample_token_id(gen_logits[pos], rng_sample=rng_sample)
                gen_x[pos] = sampled

            x = torch.cat([prompt_ids, gen_x])

            path.steps.append(
                DenoiseStep(
                    step=step_idx,
                    mean_entropy=step_entropy,
                    n_still_masked=int(masked_in_gen.sum().item()) - len(positions_to_unmask),
                    unmasked_this_step=[int(p) + prompt_len for p in positions_to_unmask],
                )
            )
            del logits, gen_logits

        path.final_tokens = x.detach().clone()
        return path

    # ── N parallel chains ──────────────────────────────────────────────────────

    def _is_dream(self) -> bool:
        """True if the loaded model is Dream-7B (not LLaDA)."""
        return getattr(self.model.config, "model_type", "").lower() == "dream"

    def _run_dream_paths(
        self,
        prompt: str,
        source_info: str,
        n_paths: int,
        gen_len: int,
        num_steps: int,
        base_seed: int = 42,
    ) -> ParallelPathResult:
        """
        Dream-7B generation via the model's own diffusion_generate() path.

        Dream's denoising schedule is model-specific and not compatible with
        the manual masked-token loop used for LLaDA. Use the tokenizer chat
        template and DreamGenerationMixin instead.
        """
        full_context = f"Retrieved context:\n{source_info}\n\nQuery:\n{prompt}\n\nAnswer:"
        msgs = [{"role": "user", "content": full_context}]
        prompt_ids_2d = self.tokenizer.apply_chat_template(
            msgs,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if hasattr(prompt_ids_2d, "input_ids"):
            prompt_ids_2d = prompt_ids_2d.input_ids
        if prompt_ids_2d.ndim == 1:
            prompt_ids_2d = prompt_ids_2d.unsqueeze(0)
        prompt_ids_2d = prompt_ids_2d.to(self.device)
        prompt_ids_1d = prompt_ids_2d[0].detach().clone()
        prompt_len = prompt_ids_2d.shape[1]

        paths: list[DenoisePath] = []
        for i in range(n_paths):
            torch.manual_seed(base_seed + i)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(base_seed + i)

            dream_steps = max(128, num_steps * 4)
            temperature = 0.1 if i == 0 else 0.3
            mask_fill = torch.full(
                (1, gen_len),
                self.mask_token_id,
                dtype=torch.long,
                device=self.device,
            )
            input_with_masks = torch.cat([prompt_ids_2d, mask_fill], dim=1)
            max_len = input_with_masks.shape[1] + 1
            generation_config = self._build_dream_generation_config(
                max_length=max_len,
                num_steps=dream_steps,
                temperature=temperature,
            )
            generation_config.mask_token_id = self.mask_token_id
            out = self.model.diffusion_generate(
                input_with_masks,
                generation_config=generation_config,
                mask_token_id=self.mask_token_id,
            )
            full_seq = out[0, : prompt_len + gen_len].detach().clone()
            path = DenoisePath(
                path_id=i,
                order=DemaskingOrder.LEARNED if i == 0 else DemaskingOrder.RANDOM,
                final_tokens=full_seq,
                steps=[],
            )
            paths.append(path)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return ParallelPathResult(prompt_tokens=prompt_ids_1d, paths=paths)

    def run_parallel_paths(
        self,
        prompt: str,
        source_info: str,
        n_paths: int = 8,
        gen_len: int = 128,
        num_steps: int = 64,
        learned_paths: int = 1,
        random_paths: int | None = None,
        hybrid_paths: int = 0,
        base_seed: int = 42,
    ) -> ParallelPathResult:
        """
        Run N denoising chains from the same masked start.

        By default:
          - 1 learned-order path (strong baseline)
          - (n_paths - 1) random-order paths  ← the novelty

        Args:
            prompt:        the RAG query (with source_info prepended)
            source_info:   retrieved passage
            n_paths:       total number of chains
            gen_len:       number of tokens to generate
            num_steps:     denoising steps (T)
            learned_paths: how many chains use learned demasking order
            random_paths:  how many use random order (default: n_paths - learned_paths)
            hybrid_paths:  how many use hybrid order
            base_seed:     reproducibility base; each chain gets base_seed + path_id
        """
        if self._is_dream():
            return self._run_dream_paths(
                prompt=prompt,
                source_info=source_info,
                n_paths=n_paths,
                gen_len=gen_len,
                num_steps=num_steps,
                base_seed=base_seed,
            )

        if random_paths is None:
            random_paths = n_paths - learned_paths - hybrid_paths

        # Build prompt ids
        prompt_str = str(prompt).strip()
        source_str = str(source_info).strip()
        if not source_str and (
            "\n" in prompt_str
            or "passage 1:" in prompt_str.lower()
            or "answer:" in prompt_str.lower()
            or "briefly answer" in prompt_str.lower()
        ):
            full_prompt = prompt_str
        else:
            full_prompt = f"Retrieved context:\n{source_str}\n\nQuery:\n{prompt_str}\n\nAnswer:"
        prompt_ids  = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            add_special_tokens=True,
        ).input_ids.squeeze(0).to(self.device)

        # Build (order, seed) pairs for all N paths
        chain_specs: list[tuple[DemaskingOrder, int]] = []
        for i in range(learned_paths):
            chain_specs.append((DemaskingOrder.LEARNED, base_seed + i))
        for i in range(random_paths):
            chain_specs.append((DemaskingOrder.RANDOM, base_seed + learned_paths + i))
        for i in range(hybrid_paths):
            chain_specs.append((DemaskingOrder.HYBRID, base_seed + learned_paths + random_paths + i))

        paths: list[DenoisePath] = []
        for path_id, (order, seed) in enumerate(chain_specs):
            path = self._run_chain(
                prompt_ids=prompt_ids,
                gen_len=gen_len,
                num_steps=num_steps,
                order=order,
                path_id=path_id,
                seed=seed,
            )
            paths.append(path)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return ParallelPathResult(prompt_tokens=prompt_ids, paths=paths)

    # ── helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_schedule(total_masked: int, num_steps: int) -> list[int]:
        """Distribute `total_masked` unmaskings across `num_steps` steps evenly."""
        base   = total_masked // num_steps
        extra  = total_masked  % num_steps
        return [base + (1 if i < extra else 0) for i in range(num_steps)]

    def decode(self, token_ids: torch.Tensor) -> str:
        ids = token_ids[(token_ids != MASK_TOKEN_ID) & (token_ids != self.mask_token_id)]
        return self.tokenizer.decode(ids, skip_special_tokens=True)
