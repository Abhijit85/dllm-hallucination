"""
LLaDA parallel denoising harness.

Wraps LLaDA-8B (masked diffusion LM) to:
  1. Run N independent denoising chains from the same masked input
  2. Support pluggable demasking-order strategies (learned | random | hybrid)
  3. Return per-token logit distributions across all steps for analysis

Model: GSAI-ML/LLaDA-8B-Instruct  (HuggingFace)
Paper: https://arxiv.org/abs/2406.11838
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


# ── constants ──────────────────────────────────────────────────────────────────
MASK_TOKEN_ID = 126336          # LLaDA's [MASK] id
MODEL_ID = "GSAI-ML/LLaDA-8B-Instruct"


class DemaskingOrder(str, Enum):
    LEARNED     = "learned"     # highest model confidence first (default)
    RANDOM      = "random"      # uniform random order
    HYBRID      = "hybrid"      # learned for first 50% steps, random for last 50%
    ENTROPY     = "entropy"     # lowest entropy (most agreed-upon) first


# ── data structures ────────────────────────────────────────────────────────────

@dataclass
class DenoiseStep:
    step: int
    x: torch.Tensor                 # (seq_len,) token ids; MASK_TOKEN_ID for still-masked
    logits: torch.Tensor            # (seq_len, vocab) raw logits
    unmasked_this_step: list[int]   # positions revealed at this step


@dataclass
class DenoisePath:
    """One complete denoising trajectory for one sample."""
    path_id: int
    order: DemaskingOrder
    steps: list[DenoiseStep] = field(default_factory=list)

    @property
    def final_tokens(self) -> torch.Tensor:
        return self.steps[-1].x

    def token_probs_at_step(self, step: int) -> torch.Tensor:
        """Softmax probs (seq_len, vocab) at a given denoising step."""
        return F.softmax(self.steps[step].logits, dim=-1)


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
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        ).to(device).eval()

    # ── core: single denoising step ────────────────────────────────────────────

    @torch.no_grad()
    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        One masked-LM forward pass.
        x:       (batch, seq_len) with MASK_TOKEN_ID for masked positions
        returns: (batch, seq_len, vocab) logits
        """
        return self.model(input_ids=x).logits

    # ── demasking-order strategies ─────────────────────────────────────────────

    @staticmethod
    def _pick_positions_to_unmask(
        x: torch.Tensor,             # (seq_len,)
        logits: torch.Tensor,        # (seq_len, vocab)
        n_to_unmask: int,
        order: DemaskingOrder,
        step: int,
        total_steps: int,
        rng: torch.Generator | None = None,
    ) -> torch.Tensor:
        """
        Returns a 1-D LongTensor of `n_to_unmask` positions to reveal.
        Only considers currently masked positions.
        """
        masked_positions = (x == MASK_TOKEN_ID).nonzero(as_tuple=True)[0]
        if len(masked_positions) == 0:
            return torch.tensor([], dtype=torch.long)
        n_to_unmask = min(n_to_unmask, len(masked_positions))

        if order == DemaskingOrder.RANDOM:
            perm = torch.randperm(len(masked_positions), generator=rng)
            return masked_positions[perm[:n_to_unmask]]

        # Confidence scores: max softmax prob at each masked position
        probs = F.softmax(logits[masked_positions], dim=-1)   # (M, vocab)
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
                perm = torch.randperm(len(masked_positions), generator=rng)
                return masked_positions[perm[:n_to_unmask]]

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
        rng = torch.Generator(device=self.device)
        if seed is not None:
            rng.manual_seed(seed)

        # Build initial x: [prompt | MASK * gen_len]
        gen_mask = torch.full((gen_len,), MASK_TOKEN_ID, dtype=torch.long, device=self.device)
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

            positions_to_unmask = self._pick_positions_to_unmask(
                x=gen_x,
                logits=gen_logits,
                n_to_unmask=n_unmask,
                order=order,
                step=step_idx,
                total_steps=num_steps,
                rng=rng,
            )

            # Sample token ids for the chosen positions
            for pos in positions_to_unmask:
                p = F.softmax(gen_logits[pos], dim=-1)
                sampled = torch.multinomial(p, num_samples=1, generator=rng)
                gen_x[pos] = sampled

            x = torch.cat([prompt_ids, gen_x])

            path.steps.append(
                DenoiseStep(
                    step=step_idx,
                    x=x.clone(),
                    logits=logits.detach().clone(),
                    unmasked_this_step=[int(p) + prompt_len for p in positions_to_unmask],
                )
            )

        return path

    # ── N parallel chains ──────────────────────────────────────────────────────

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
        if random_paths is None:
            random_paths = n_paths - learned_paths - hybrid_paths

        # Build prompt ids
        full_prompt = f"Retrieved context:\n{source_info}\n\nQuery:\n{prompt}\n\nAnswer:"
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

        return ParallelPathResult(prompt_tokens=prompt_ids, paths=paths)

    # ── helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_schedule(total_masked: int, num_steps: int) -> list[int]:
        """Distribute `total_masked` unmaskings across `num_steps` steps evenly."""
        base   = total_masked // num_steps
        extra  = total_masked  % num_steps
        return [base + (1 if i < extra else 0) for i in range(num_steps)]

    def decode(self, token_ids: torch.Tensor) -> str:
        ids = token_ids[token_ids != MASK_TOKEN_ID]
        return self.tokenizer.decode(ids, skip_special_tokens=True)
