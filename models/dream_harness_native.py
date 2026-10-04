"""
models/dream_harness_native.py
------------------------------
Dream-7B harness for PaRaDe — Option 1: N independent diffusion_generate() calls.

APPROACH
--------
Instead of a manual denoising loop (which fails due to training-inference mismatch),
run Dream's own diffusion_generate() N times with different random seeds and
temperature > 0. Each call produces one complete coherent answer via Dream's
trained generation pipeline. Cross-answer diversity gives the hallucination signal.

WHY THIS WORKS
--------------
Dream's diffusion_generate() uses the exact same attention mask, logit shift,
and stochastic schedule as during training. Running it N times with temperature > 0
gives different answers for uncertain questions (high cross-path entropy =
hallucination) and consistent answers for confident questions (low entropy = grounded).

WHY MANUAL LOOPS FAILED
------------------------
Dream was trained with full-sequence diffusion (both prompt + answer tokens noised).
Freezing prompt tokens and denoising only the answer creates a distribution mismatch.
diffusion_generate() handles this correctly internally.

KEY FIXES APPLIED
-----------------
1. mask_token_id must be 151666 (not LLaDA's 126336)
2. generation_config.json has mask_token_id: null — must pass explicitly
3. NaN after logit shift — suppressed via nan_to_num in logits hook
4. EOS/PAD (151643) suppressed via generation_logits_hook_func
5. Pre-fill generation region with real mask tokens before calling diffusion_generate
   so Dream's F.pad adds zero extra tokens (max_length = input.shape[1] + 1)
"""

from __future__ import annotations

import glob
import importlib.util
import os

import torch
from transformers import AutoModel, AutoTokenizer

from models.llada_harness import (
    DemaskingOrder,
    DenoisePath,
    ParallelPathResult,
    _patch_llada_transformers_compat,
)

DREAM_MASK_ID = 151666
DREAM_EOS_ID = 151643
DREAM_PAD_ID = 151643


def _find_snap() -> str:
    """Locate Dream-7B snapshot on this server."""
    patterns = [
        "/mnt/shared/shared_hf_home/hub/models--Dream-org--Dream-v0-Instruct-7B"
        "/snapshots/*/modeling_dream.py",
        "/home/**/.cache/huggingface/hub/models--Dream-org*"
        "/snapshots/*/modeling_dream.py",
    ]
    for pat in patterns:
        hits = glob.glob(pat, recursive=True)
        if hits:
            return os.path.dirname(hits[0])
    raise FileNotFoundError("Dream-7B snapshot not found. Check mount paths.")


def _load_dream_gen_utils(snap: str):
    """Load DreamGenerationConfig from Dream's remote-code snapshot."""
    path = os.path.join(snap, "generation_utils.py")
    spec = importlib.util.spec_from_file_location("dream_gen_utils", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class DreamHarnessNative:
    """
    Dream-7B harness for PaRaDe using N independent diffusion_generate() calls.

    Produces a ParallelPathResult with the same interface as LLaDAHarness.
    Each of the N paths is one complete Dream generation with a different seed
    and temperature=0.3, giving genuine cross-path diversity.
    """

    def __init__(self, model_id: str | None = None, device: str = "cuda"):
        self.device = device
        self.mask_token_id = DREAM_MASK_ID

        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        _patch_llada_transformers_compat()

        snap = model_id if (model_id and os.path.isdir(model_id)) else _find_snap()
        self._snap = snap
        print(f"Dream snapshot: {snap}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            snap, trust_remote_code=True, local_files_only=True
        )
        self.model = (
            AutoModel.from_pretrained(
                snap,
                trust_remote_code=True,
                local_files_only=True,
                torch_dtype=torch.bfloat16,
            )
            .to(device)
            .eval()
        )

        gu = _load_dream_gen_utils(snap)
        self._DreamGenerationConfig = gu.DreamGenerationConfig

        print(f"  mask_token_id : {self.mask_token_id}")

        # Logits hook: suppress EOS/PAD/mask and sanitize NaN at every step
        _suppress = [DREAM_MASK_ID, DREAM_EOS_ID]

        def _hook(step, x, logits):
            if logits is None:
                return logits
            logits = torch.nan_to_num(logits, nan=-1e9, posinf=1e9, neginf=-1e9)
            for sid in _suppress:
                logits[:, :, sid] = -1e9
            return logits

        self._logits_hook = _hook

    # ── prompt ────────────────────────────────────────────────────────────────

    def _build_prompt_ids(self, prompt: str, source_info: str) -> torch.Tensor:
        """2D prompt tensor (1, prompt_len) on device."""
        prompt_text = prompt.strip()
        if source_info and source_info.strip():
            content = (
                f"Context: {source_info.strip()[:600]}\n\n"
                f"Question: {prompt_text}\n\nAnswer:"
            )
        elif "\n" in prompt_text or "output:" in prompt_text.lower():
            # RAGTruth already stores a fully formatted task prompt. Re-wrapping
            # it as Question/Answer degrades Dream generations badly.
            content = prompt_text
        else:
            content = f"Question: {prompt_text}\n\nAnswer:"

        msgs = [{"role": "user", "content": content}]
        ids_2d = self.tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt"
        )
        if hasattr(ids_2d, "input_ids"):
            ids_2d = ids_2d.input_ids
        if ids_2d.ndim == 1:
            ids_2d = ids_2d.unsqueeze(0)
        return ids_2d.to(self.device)

    # ── single path via diffusion_generate() ─────────────────────────────────

    def _run_one_path(
        self,
        prompt_ids_2d: torch.Tensor,  # (1, prompt_len)
        gen_len: int,
        dream_steps: int,
        temperature: float,
        path_id: int,
        seed: int,
    ) -> DenoisePath:
        """
        One Dream generation path using diffusion_generate().

        Pre-fills generation region with mask tokens (151666) so Dream's
        internal F.pad adds nothing (max_length = input.shape[1] + 1 satisfies
        Dream's validator: requires input_len < max_length strictly).

        Output is trimmed to exact (prompt_len + gen_len) tokens.
        """
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        prompt_len = prompt_ids_2d.shape[1]

        # Pre-fill generation region — avoids F.pad(value=None) → token 0 bug
        mask_fill = torch.full(
            (1, gen_len), self.mask_token_id, dtype=torch.long, device=self.device
        )
        x = torch.cat([prompt_ids_2d, mask_fill], dim=1)  # (1, prompt+gen)

        cfg = self._DreamGenerationConfig(
            max_length=x.shape[1] + 1,  # +1 satisfies Dream's validator
            steps=dream_steps,
            temperature=temperature,
            alg="origin",
            eps=0.001,
            mask_token_id=self.mask_token_id,
            pad_token_id=DREAM_PAD_ID,
            eos_token_id=DREAM_EOS_ID,
            output_history=False,
        )

        out = self.model.diffusion_generate(
            x,
            generation_config=cfg,
            mask_token_id=self.mask_token_id,
            generation_logits_hook_func=self._logits_hook,
        )

        # Trim to exact length — discard the +1 validator token
        full_seq = out[0, : prompt_len + gen_len].cpu()

        return DenoisePath(
            path_id=path_id,
            order=DemaskingOrder.LEARNED if path_id == 0 else DemaskingOrder.RANDOM,
            final_tokens=full_seq,
            steps=[],  # no step data from diffusion_generate
        )

    # ── public API ────────────────────────────────────────────────────────────

    @staticmethod
    def _build_schedule(total: int, steps: int) -> list[int]:
        """Distribute total unmaskings evenly across steps (matches LLaDA interface)."""
        base, extra = divmod(total, steps)
        return [base + (1 if i < extra else 0) for i in range(steps)]

    def forward_single(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Single forward pass with Dream's correct tok_idx + 4D attention mask.
        Returns logits (1, seq_len, vocab) with Dream's logit shift applied.

        Used by _self_verify() in run_detection_auroc.py for forced-choice
        verification. Does NOT require a denoising loop — just one forward pass.

        Verified working: fc_no_frac gap = 0.5265 (Paris=0.058, Berlin=0.585)
        """
        seq_len = input_ids.shape[1]
        attn_1d = torch.ones(1, seq_len, dtype=torch.float, device=self.device)
        tok_idx = attn_1d.long().cumsum(-1) - 1
        attn_4d = torch.logical_and(
            attn_1d.unsqueeze(1).unsqueeze(-2),
            attn_1d.unsqueeze(1).unsqueeze(-1),
        )
        with torch.no_grad():
            out = self.model(input_ids, attn_4d, tok_idx)
            logits = out.logits
            logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
            logits = torch.nan_to_num(logits, nan=-1e9, posinf=1e9, neginf=-1e9)
        return logits

    def remask_and_refine(
        self,
        prompt_ids_2d: torch.Tensor,
        generated_tokens: torch.Tensor,
        high_entropy_mask: torch.Tensor,
        gen_len: int,
        dream_steps: int = 512,
    ) -> torch.Tensor:
        """
        Pass 3 for Dream: remask high-entropy positions, re-denoise with source.

        Dream's full-sequence training objective handles partially-masked inputs
        naturally. Confident tokens (non-masked) anchor generation of uncertain
        ones. This replicates LLaDA's targeted remasking using Dream's own
        diffusion_generate() pipeline.

        Args:
            prompt_ids_2d:     (1, prompt_len) — includes source context
            generated_tokens:  (gen_len,) cpu — majority vote from Pass 1
            high_entropy_mask: (gen_len,) bool — True = remask this position
            gen_len:           expected output length
            dream_steps:       denoising steps (more = better refinement)

        Returns:
            (gen_len,) tensor — refined answer tokens
        """
        n_remasked = int(high_entropy_mask.sum().item())
        if n_remasked == 0:
            return generated_tokens

        # Remask uncertain positions, keep confident ones as anchors
        refined = generated_tokens.to(self.device).clone()
        refined[high_entropy_mask.to(self.device)] = self.mask_token_id

        # Build: [prompt | partially_masked_answer]
        x = torch.cat(
            [
                prompt_ids_2d[0],
                refined,
            ]
        ).unsqueeze(
            0
        )  # (1, prompt_len + gen_len)

        cfg = self._DreamGenerationConfig(
            max_length=x.shape[1] + 1,
            steps=dream_steps,
            temperature=0.0,  # greedy for refinement
            alg="origin",
            eps=0.001,
            mask_token_id=self.mask_token_id,
            pad_token_id=DREAM_EOS_ID,
            eos_token_id=DREAM_EOS_ID,
            output_history=False,
        )

        out = self.model.diffusion_generate(
            x,
            generation_config=cfg,
            mask_token_id=self.mask_token_id,
            generation_logits_hook_func=self._logits_hook,
        )

        plen = prompt_ids_2d.shape[1]
        return out[0, plen : plen + gen_len].cpu()

    def compute_entropy_mask(
        self,
        result: ParallelPathResult,
        prompt_len: int,
        top_k_percent: float = 0.20,
    ) -> torch.Tensor:
        """
        Identify top-k% highest-entropy positions in generation region.
        Returns bool tensor (gen_len,) — True = high entropy = remask.
        Mirrors LLaDA's compute_disagreement top-k selection.
        """
        entropy = result.token_entropy()  # (full_seq_len,)
        gen_entropy = entropy[prompt_len:]  # (gen_len,)
        k = max(1, int(len(gen_entropy) * top_k_percent))
        threshold = gen_entropy.topk(k).values.min()
        return gen_entropy >= threshold

    def compute_first_step_entropy(
        self,
        prompt: str,
        source_info: str,
        gen_len: int = 64,
    ) -> torch.Tensor:
        """
        Compute per-token Shannon entropy at the first denoising step.

        Equivalent to LLaDA's traj_early_entropy — but implemented via a
        single forward_single() call on a fully masked answer region.

        Mechanism:
        - Start with [prompt | MASK*gen_len] — no tokens committed yet
        - Run one bidirectional forward pass (forward_single)
        - Read softmax entropy at each masked position in the answer region
        - High entropy = model uncertain at first step = likely hallucination

        This captures uncertainty BEFORE commitment, unlike fc_no_frac which
        reads uncertainty AFTER a full answer is generated. For multi-hop QA
        where Dream generates confident-but-wrong answers, first-step entropy
        may be more discriminative than post-generation verification.

        Returns:
            (gen_len,) float tensor of per-token entropy values (cpu)
        """
        prompt_ids = self._build_prompt_ids(prompt, source_info)
        plen = prompt_ids.shape[1]

        # Fully masked answer region — no tokens committed
        masked = torch.full(
            (1, gen_len), self.mask_token_id, dtype=torch.long, device=self.device
        )
        full_ids = torch.cat([prompt_ids, masked], dim=1)  # (1, plen+gen_len)

        # Single bidirectional forward pass
        logits = self.forward_single(full_ids)  # (1, plen+gen_len, vocab)

        # Entropy at each masked position in the answer region only
        gen_logits = logits[0, plen : plen + gen_len, :]  # (gen_len, vocab)
        probs = torch.softmax(gen_logits.float(), dim=-1)
        # Numerically stable Shannon entropy
        entropy = -(probs * (probs.clamp(min=1e-10)).log()).sum(dim=-1)  # (gen_len,)

        return entropy.cpu()

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
        Run N independent Dream diffusion_generate() calls.

        Path 0: temperature=0.0 (greedy deterministic — 'learned' equivalent)
        Paths 1..N-1: temperature=0.3 (stochastic — 'random' equivalent)

        dream_steps = max(256, num_steps * 8) — Dream needs many more steps
        than LLaDA for quality generation.
        """
        dream_steps = max(256, num_steps * 8)

        prompt_ids_2d = self._build_prompt_ids(prompt, source_info)
        prompt_ids_1d = prompt_ids_2d[0].cpu()

        paths: list[DenoisePath] = []
        for i in range(n_paths):
            # Path 0: deterministic (greedy). Paths 1+: stochastic.
            temperature = 0.0 if i == 0 else 0.3
            path = self._run_one_path(
                prompt_ids_2d=prompt_ids_2d,
                gen_len=gen_len,
                dream_steps=dream_steps,
                temperature=temperature,
                path_id=i,
                seed=base_seed + i,
            )
            paths.append(path)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return ParallelPathResult(
            prompt_tokens=prompt_ids_1d,
            paths=paths,
        )

    def run_full_pipeline(
        self,
        prompt: str,
        source_info: str,
        n_paths: int = 8,
        gen_len: int = 32,
        num_steps: int = 32,
        fc_threshold: float = 0.5,
        top_k_percent: float = 0.20,
        refine_steps: int = 512,
        base_seed: int = 42,
        **kwargs,
    ) -> dict:
        """
        Full three-pass PaRaDe pipeline for Dream-7B.

        Pass 1: N parallel diffusion_generate() calls → cross-path entropy
        Pass 2: forced-choice verification via forward_single() → fc_no_frac
        Pass 3: targeted remasking of high-entropy positions → corrected answer

        Returns dict with:
            result:         ParallelPathResult from Pass 1
            answer_before:  majority vote answer before remasking
            answer_after:   refined answer after remasking (or same if not triggered)
            fc_no_frac:     forced-choice verification score
            remasked:       True if remasking was applied
            n_remasked:     number of positions remasked
        """
        prompt_ids_2d = self._build_prompt_ids(prompt, source_info)
        prompt_len = prompt_ids_2d.shape[1]

        # ── Pass 1: parallel generation ───────────────────────────────────────
        result = self.run_parallel_paths(
            prompt=prompt,
            source_info=source_info,
            n_paths=n_paths,
            gen_len=gen_len,
            num_steps=num_steps,
            base_seed=base_seed,
        )
        mv = result.majority_vote()
        gen_tokens = mv[prompt_len:]  # (gen_len,)
        answer_before = self.decode(gen_tokens)

        # ── Pass 2: forced-choice verification ───────────────────────────────
        stem = (
            f"Context: {source_info.strip()[:400]}\n\n"
            f"Q: {prompt.strip()}\nA: {answer_before.strip()}\n"
            f"Is this answer correct?"
        )
        stem_ids = self.tokenizer.encode(
            stem, add_special_tokens=True, return_tensors="pt"
        ).to(self.device)
        mask_tok = torch.tensor([[self.mask_token_id]], device=self.device)
        full_ids = torch.cat([stem_ids, mask_tok], dim=1)

        logits = self.forward_single(full_ids)
        mask_logits = logits[0, -1, :]
        probs = torch.softmax(mask_logits.float(), dim=-1)

        def best_prob(words):
            best = 0.0
            for w in words:
                ids = self.tokenizer.encode(w, add_special_tokens=False)
                if ids:
                    best = max(best, probs[ids[0]].item())
            return best

        p_yes = best_prob([" Yes", "Yes", " yes", "yes"])
        p_no = best_prob([" No", "No", " no", "no"])
        fc = p_no / (p_yes + p_no + 1e-10)

        # ── Pass 3: targeted remasking if fc_no_frac is high ─────────────────
        remasked = False
        n_remasked = 0
        answer_after = answer_before

        if fc >= fc_threshold:
            entropy_mask = self.compute_entropy_mask(result, prompt_len, top_k_percent)
            n_remasked = int(entropy_mask.sum().item())

            if n_remasked > 0:
                refined_tokens = self.remask_and_refine(
                    prompt_ids_2d=prompt_ids_2d,
                    generated_tokens=gen_tokens.cpu(),
                    high_entropy_mask=entropy_mask,
                    gen_len=gen_len,
                    dream_steps=refine_steps,
                )
                answer_after = self.decode(refined_tokens)
                remasked = True

        return {
            "result": result,
            "answer_before": answer_before,
            "answer_after": answer_after,
            "fc_no_frac": fc,
            "p_yes": p_yes,
            "p_no": p_no,
            "remasked": remasked,
            "n_remasked": n_remasked,
        }

    def decode(self, token_ids: torch.Tensor) -> str:
        ids = token_ids[token_ids != self.mask_token_id]
        return self.tokenizer.decode(ids.tolist(), skip_special_tokens=True)


# ── standalone smoke test ─────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== DreamHarnessNative full pipeline smoke test ===\n")
    h = DreamHarnessNative()

    # Test correct answer — fc_no_frac should be LOW (model confident)
    # Test wrong answer  — fc_no_frac should be HIGH (model doubts)
    print("--- Pass 2: forced-choice verification ---")
    for answer, label in [("Paris", "CORRECT"), ("Berlin", "WRONG")]:
        stem = (
            f"Q: What is the capital of France?\nA: {answer}\nIs this answer correct?"
        )
        ids = h.tokenizer.encode(stem, add_special_tokens=True, return_tensors="pt").to(
            h.device
        )
        mask = torch.tensor([[h.mask_token_id]], device=h.device)
        full = torch.cat([ids, mask], dim=1)
        logits = h.forward_single(full)
        probs = torch.softmax(logits[0, -1, :].float(), dim=-1)
        p_yes = max(
            probs[h.tokenizer.encode(w, add_special_tokens=False)[0]].item()
            for w in [" Yes", "Yes", "yes"]
            if h.tokenizer.encode(w, add_special_tokens=False)
        )
        p_no = max(
            probs[h.tokenizer.encode(w, add_special_tokens=False)[0]].item()
            for w in [" No", "No", "no"]
            if h.tokenizer.encode(w, add_special_tokens=False)
        )
        fc = p_no / (p_yes + p_no + 1e-10)
        ok = (label == "CORRECT" and fc < 0.5) or (label == "WRONG" and fc > 0.5)
        print(
            f"  [{label}] answer={answer!r:8s} fc_no_frac={fc:.4f} {'OK' if ok else 'FAIL'}"
        )
    print()

    # Test full pipeline on one QA pair
    print("--- Full pipeline (Pass 1 + 2 + 3) ---")
    out = h.run_full_pipeline(
        prompt="What is the capital of France?",
        source_info="France is a country in Western Europe. Its capital is Paris.",
        n_paths=4,
        gen_len=16,
        num_steps=32,
        refine_steps=256,
    )
    print(f"  Answer before remasking: {out['answer_before'][:60]!r}")
    print(f"  fc_no_frac:              {out['fc_no_frac']:.4f}")
    print(
        f"  Remasking triggered:     {out['remasked']} ({out['n_remasked']} positions)"
    )
    print(f"  Answer after remasking:  {out['answer_after'][:60]!r}")
    print()
    print("Smoke test complete.")
