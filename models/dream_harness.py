"""
models/dream_harness.py
-----------------------
Dream-7B-Instruct harness for PaRaDe — full parallel denoising implementation.

WHY A SEPARATE HARNESS
----------------------
Dream-7B uses a fundamentally different denoising contract than LLaDA:

1. LOGIT SHIFT: Dream applies a one-position left shift to all logits before
   sampling — logits[i] predicts the token that should fill position i+1.
   Without this shift, every masked position uses its own (wrong) logit.
   This is unique to Dream's architecture and absent in LLaDA.

2. ATTENTION MASK: Dream computes tok_idx = cumsum(attention_mask) - 1 to
   give each token its true position ID (skipping padding). The attention
   mask is reshaped to [B,1,N,N] for 2D causal-like masking. LLaDA uses
   standard causal masking.

3. FORWARD SIGNATURE: Dream's forward() takes (input_ids, attention_mask,
   position_ids) as positional args. LLaDA uses keyword args only.

4. PROMPT FORMAT: Dream requires Qwen2.5 chat template. LLaDA uses plain text.

PaRaDe requires step-level control (custom reveal orders, per-step entropy).
Dream's diffusion_generate() is a black box — we reimplement its _sample()
loop with PaRaDe's N-path exploration grafted in.

ICLR PAPER NOTE
---------------
This harness enables multi-backbone evaluation on Dream-7B-Instruct alongside
LLaDA-8B-Instruct, matching the evaluation setup of TraceDet, TDGNet, DynHD.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from models.llada_harness import (
    DemaskingOrder,
    DenoisePath,
    DenoiseStep,
    ParallelPathResult,
    _patch_llada_transformers_compat,
    resolve_local_llada_model,
)

DREAM_MASK_ID = 151666
DREAM_EOS_ID = 151643
DREAM_PAD_ID = 151643
DREAM_BOS_ID = 151643
DREAM_EPS = 0.001


class DreamHarness:
    """
    Dream-7B-Instruct harness for PaRaDe parallel denoising.

    Reimplements Dream's _sample() loop with:
    - PaRaDe's N-path exploration (learned + random reveal orders)
    - Per-step entropy tracking for H3 trajectory analysis
    - Dream's exact logit shift and tok_idx mechanics
    - Qwen2.5 chat template prompt formatting
    """

    def __init__(
        self,
        model_id: str = "Dream-org/Dream-v0-Instruct-7B",
        device: str = "cuda",
    ):
        self.model_id = model_id
        self.model_path = resolve_local_llada_model(model_id)
        self.device = device
        self.mask_token_id = DREAM_MASK_ID
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        _patch_llada_transformers_compat()

        print(f"Loading Dream-7B tokenizer: {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            local_files_only=True,
        )

        print(f"Loading Dream-7B model: {self.model_path}")
        self.model = (
            AutoModel.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                local_files_only=True,
            )
            .to(device)
            .eval()
        )

        print(f"  mask_token_id : {self.mask_token_id}")
        print(f"  device        : {device}")

    # ── prompt ────────────────────────────────────────────────────────────────

    def _build_prompt_ids(self, prompt: str, source_info: str) -> torch.Tensor:
        """
        Build tokenized prompt using Qwen2.5 chat template.
        Returns 1D tensor on self.device.
        """
        if source_info and source_info.strip():
            content = (
                f"Context: {source_info.strip()[:600]}\n\n"
                f"Question: {prompt.strip()}\n\nAnswer:"
            )
        else:
            content = f"Question: {prompt.strip()}\n\nAnswer:"

        msgs = [{"role": "user", "content": content}]
        ids_2d = self.tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt"
        )
        if hasattr(ids_2d, "input_ids"):
            ids_2d = ids_2d.input_ids
        ids_2d = ids_2d.to(self.device)
        return ids_2d.squeeze(0)

    # ── Dream forward pass ────────────────────────────────────────────────────

    def _dream_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        One Dream forward pass with proper attention mask and tok_idx.

        Dream's forward signature: (input_ids, attention_mask, position_ids)
        attention_mask = "full" -> no padding, standard bidirectional attention
        tok_idx = None          -> use default position IDs

        Returns shifted logits: (seq_len, vocab)
        The shift logits[i] -> correct prediction for position i by applying
        Dream's one-position left shift.
        """
        x_2d = x.unsqueeze(0)

        with torch.no_grad():
            out = self.model(x_2d, "full", None)

        logits = out.logits.squeeze(0)

        # Dream uses next-position logits; shift so logits[i] scores token for i.
        logits = torch.cat([logits[:1], logits[:-1]], dim=0)

        return logits

    @torch.no_grad()
    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        LLaDA-compatible batched forward used by refinement code.

        `parallel_remask` calls `_forward(batch)` on the active harness.
        Dream's native helper is sequence-oriented, so wrap it here and keep
        the return shape aligned with LLaDAHarness: (batch, seq_len, vocab).
        """
        if x.ndim == 1:
            return self._dream_forward(x).unsqueeze(0)
        if x.ndim != 2:
            raise ValueError(
                f"DreamHarness._forward expects 1D or 2D input, got {tuple(x.shape)}"
            )
        return torch.stack([self._dream_forward(row) for row in x], dim=0)

    # ── single denoising chain ────────────────────────────────────────────────

    def _run_chain(
        self,
        prompt_ids: torch.Tensor,
        gen_len: int,
        num_steps: int,
        order: DemaskingOrder,
        path_id: int,
        seed: int,
    ) -> DenoisePath:
        """
        One Dream denoising chain with PaRaDe-style reveal order.

        Mirrors Dream's _sample() but replaces the stochastic p_transfer
        schedule with PaRaDe's deterministic reveal strategies.
        """
        rng_cpu = torch.Generator(device="cpu")
        rng_cpu.manual_seed(seed)
        rng_sample = torch.Generator(device=self.device)
        rng_sample.manual_seed(seed)

        prompt_len = len(prompt_ids)
        total_masked = gen_len

        gen_mask = torch.full(
            (gen_len,), self.mask_token_id, dtype=torch.long, device=self.device
        )
        x = torch.cat([prompt_ids, gen_mask])

        unmask_schedule = self._build_schedule(total_masked, num_steps)

        path = DenoisePath(path_id=path_id, order=order)

        for step_idx, n_unmask in enumerate(unmask_schedule):
            logits = self._dream_forward(x)
            gen_logits = logits[prompt_len:]
            gen_x = x[prompt_len:]

            for sid in [self.mask_token_id, DREAM_EOS_ID, DREAM_PAD_ID]:
                gen_logits[:, sid] = -1e9

            masked_in_gen = gen_x == self.mask_token_id
            if masked_in_gen.any():
                probs_masked = F.softmax(gen_logits[masked_in_gen], dim=-1)
                step_entropy = float(
                    -(probs_masked * (probs_masked + 1e-10).log()).sum(-1).mean()
                )
            else:
                step_entropy = 0.0

            positions = self._pick_positions(
                gen_x=gen_x,
                gen_logits=gen_logits,
                n_unmask=n_unmask,
                order=order,
                rng=rng_cpu,
            )

            for pos in positions:
                p = self._safe_softmax(gen_logits[pos])
                sampled = torch.multinomial(p, num_samples=1, generator=rng_sample)
                gen_x = gen_x.clone()
                gen_x[pos] = sampled
                x = torch.cat([prompt_ids, gen_x])

            path.steps.append(
                DenoiseStep(
                    step=step_idx,
                    mean_entropy=step_entropy,
                    n_still_masked=int(masked_in_gen.sum()) - len(positions),
                    unmasked_this_step=[int(p) + prompt_len for p in positions],
                )
            )

            del logits
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        path.final_tokens = x.cpu()
        return path

    def _pick_positions(
        self,
        gen_x: torch.Tensor,
        gen_logits: torch.Tensor,
        n_unmask: int,
        order: DemaskingOrder,
        rng: torch.Generator,
    ) -> list[int]:
        """Pick which masked positions to reveal this step."""
        masked = (gen_x == self.mask_token_id).nonzero(as_tuple=True)[0]
        if len(masked) == 0:
            return []
        n_unmask = min(n_unmask, len(masked))

        if order == DemaskingOrder.RANDOM:
            perm = torch.randperm(len(masked), generator=rng)
            return masked[perm[:n_unmask]].tolist()

        probs = self._safe_softmax(gen_logits[masked])
        confidence = probs.max(dim=-1).values

        if order == DemaskingOrder.LEARNED:
            top_k = confidence.topk(n_unmask).indices
            return masked[top_k].tolist()

        if order == DemaskingOrder.ENTROPY:
            ent = -(probs * (probs + 1e-10).log()).sum(-1)
            top_k = ent.topk(n_unmask, largest=False).indices
            return masked[top_k].tolist()

        raise ValueError(f"Unsupported order for Dream: {order}")

    def _sample_token_id(
        self,
        logits: torch.Tensor,
        rng_sample: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample one token id while suppressing Dream mask/special ids."""
        masked_logits = logits.clone()
        for special_id in [
            self.mask_token_id,
            DREAM_EOS_ID,
            DREAM_PAD_ID,
            DREAM_BOS_ID,
        ]:
            if 0 <= special_id < masked_logits.shape[-1]:
                masked_logits[special_id] = -1e9

        probs = self._safe_softmax(masked_logits)
        try:
            return torch.multinomial(probs, num_samples=1, generator=rng_sample)
        except RuntimeError:
            return probs.argmax(dim=-1, keepdim=True)

    def _pick_positions_to_unmask(
        self,
        x: torch.Tensor,
        logits: torch.Tensor,
        n_to_unmask: int,
        order: DemaskingOrder,
        step: int,
        total_steps: int,
        rng_cpu: torch.Generator | None = None,
    ) -> torch.Tensor:
        """
        LLaDA-compatible position picker used by refinement code.

        Dream supports learned, random, and entropy order directly. For hybrid
        refinement, mirror the LLaDA behavior: early learned, late random.
        """
        if order == DemaskingOrder.HYBRID:
            midpoint = total_steps // 2
            effective_order = (
                DemaskingOrder.LEARNED if step < midpoint else DemaskingOrder.RANDOM
            )
        else:
            effective_order = order

        positions = self._pick_positions(
            gen_x=x,
            gen_logits=logits,
            n_unmask=n_to_unmask,
            order=effective_order,
            rng=rng_cpu if rng_cpu is not None else torch.Generator(device="cpu"),
        )
        return torch.tensor(positions, dtype=torch.long, device=x.device)

    # ── schedule ──────────────────────────────────────────────────────────────

    @staticmethod
    def _build_schedule(total_masked: int, num_steps: int) -> list[int]:
        """Distribute total_masked unmaskings evenly across num_steps."""
        base = total_masked // num_steps
        extra = total_masked % num_steps
        return [base + (1 if i < extra else 0) for i in range(num_steps)]

    @staticmethod
    def _safe_softmax(logits: torch.Tensor) -> torch.Tensor:
        """Numerically safe softmax with argmax fallback for invalid rows."""
        safe_logits = torch.nan_to_num(
            logits.float(), nan=-1e4, posinf=1e4, neginf=-1e4
        )
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

    # ── public API ────────────────────────────────────────────────────────────

    def run_parallel_paths(
        self,
        prompt: str,
        source_info: str,
        n_paths: int = 8,
        gen_len: int = 64,
        num_steps: int = 32,
        learned_paths: int = 1,
        base_seed: int = 42,
        **kwargs,
    ) -> ParallelPathResult:
        """
        Run N Dream denoising chains with different reveal orders.

        Same interface as LLaDAHarness.run_parallel_paths().
        """
        del kwargs
        dream_steps = max(64, num_steps * 2)
        random_paths = n_paths - learned_paths

        prompt_ids = self._build_prompt_ids(prompt, source_info)

        chain_specs = []
        for i in range(learned_paths):
            chain_specs.append((DemaskingOrder.LEARNED, base_seed + i))
        for i in range(random_paths):
            chain_specs.append((DemaskingOrder.RANDOM, base_seed + learned_paths + i))

        paths: list[DenoisePath] = []
        for path_id, (order, seed) in enumerate(chain_specs):
            path = self._run_chain(
                prompt_ids=prompt_ids,
                gen_len=gen_len,
                num_steps=dream_steps,
                order=order,
                path_id=path_id,
                seed=seed,
            )
            paths.append(path)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return ParallelPathResult(
            prompt_tokens=prompt_ids.cpu(),
            paths=paths,
        )

    def decode(self, token_ids: torch.Tensor) -> str:
        """Decode token ids, filtering mask and special tokens."""
        ids = token_ids[token_ids != self.mask_token_id]
        return self.tokenizer.decode(ids.tolist(), skip_special_tokens=True)


if __name__ == "__main__":

    print("=== DreamHarness standalone smoke test ===\n")
    h = DreamHarness()

    test_cases = [
        ("What is the capital of France?", "France is a country in Western Europe."),
        ("Who wrote Hamlet?", "Hamlet is a tragedy by William Shakespeare."),
        (
            "What does DNA stand for?",
            "DNA carries genetic information in living organisms.",
        ),
    ]

    for question, context in test_cases:
        result = h.run_parallel_paths(
            prompt=question,
            source_info=context,
            n_paths=4,
            gen_len=32,
            num_steps=32,
        )
        plen = len(result.prompt_tokens)
        mv = result.majority_vote()[plen:]
        text = h.decode(mv)
        toks = mv.tolist()[:8]
        uniq = len(set(toks))

        print(f"Q:       {question}")
        print(f"Tokens:  {toks}  (unique: {uniq}/8)")
        print(f"Answer:  {text[:80]!r}")
        print(f"Status:  {'OK - real tokens' if uniq > 2 else 'DEGENERATE'}")
        print()

    print("=== Entropy check (N=4 paths) ===")
    result2 = h.run_parallel_paths(
        prompt="The capital of France is",
        source_info="",
        n_paths=4,
        gen_len=16,
        num_steps=32,
    )
    plen2 = len(result2.prompt_tokens)
    from strategies.parallel_remask import compute_disagreement

    report = compute_disagreement(result2, top_k_percent=0.20)
    ent = report.token_entropy[plen2:]
    print(f"Mean entropy over gen tokens: {ent.mean():.4f}")
    print(f"Max entropy:                  {ent.max():.4f}")
    print("(non-zero entropy = paths disagree = signal exists)")
