"""
scripts/test_dream_official.py
------------------------------
Tests Dream-7B using its exact official inference path from the GitHub repo.
https://github.com/HKUNLP/Dream

Run BEFORE any harness integration:
    CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/test_dream_official.py

This script has four tests in increasing complexity:
  TEST 1: Exact GitHub repo example (open-ended generation)
  TEST 2: Simple QA without context
  TEST 3: QA with context
  TEST 4: Completion-style prompt
"""

import glob
import importlib.util
import os
import sys

import torch

sys.path.insert(0, ".")

from models.llada_harness import _patch_llada_transformers_compat

POSSIBLE_SNAPS = glob.glob(
    "/mnt/shared/shared_hf_home/hub/models--Dream-org--Dream-v0-Instruct-7B"
    "/snapshots/*/modeling_dream.py"
)
if not POSSIBLE_SNAPS:
    POSSIBLE_SNAPS = glob.glob(
        "/home/**/.cache/huggingface/hub/models--Dream-org*"
        "/snapshots/*/modeling_dream.py",
        recursive=True,
    )

if not POSSIBLE_SNAPS:
    print("ERROR: Dream snapshot not found. Check mount paths.")
    sys.exit(1)

SNAP = os.path.dirname(POSSIBLE_SNAPS[0])
print(f"Using Dream snapshot: {SNAP}\n")

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
_patch_llada_transformers_compat()

from transformers import AutoModel, AutoTokenizer  # noqa: E402

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    SNAP, trust_remote_code=True, local_files_only=True
)

print("Loading model...")
model = AutoModel.from_pretrained(
    SNAP,
    trust_remote_code=True,
    local_files_only=True,
    torch_dtype=torch.bfloat16,
).cuda().eval()

print(f"mask_token_id: {model.config.mask_token_id}")
print(f"eos_token_id:  {tokenizer.eos_token_id}")
print()

spec = importlib.util.spec_from_file_location(
    "dream_gen_utils", os.path.join(SNAP, "generation_utils.py")
)
gu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gu)
DreamGenerationConfig = gu.DreamGenerationConfig

MASK_ID = model.config.mask_token_id
EOS_ID = tokenizer.eos_token_id


def decode(ids):
    ids = [i for i in ids if i not in (MASK_ID, EOS_ID)]
    return tokenizer.decode(ids, skip_special_tokens=True)


def run(label, messages, max_new_tokens=32, steps=512, temperature=0.3):
    ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    )
    if hasattr(ids, "input_ids"):
        ids = ids.input_ids
    ids = ids.to("cuda")

    mask_fill = torch.full(
        (1, max_new_tokens), MASK_ID, dtype=torch.long, device="cuda"
    )
    x = torch.cat([ids, mask_fill], dim=1)

    cfg = DreamGenerationConfig(
        max_length=x.shape[1] + 1,
        steps=steps,
        temperature=temperature,
        alg="origin",
        eps=0.001,
        mask_token_id=MASK_ID,
        pad_token_id=EOS_ID,
        eos_token_id=EOS_ID,
    )

    def suppress_hook(step, x, logits):
        del step, x
        if logits is not None:
            logits = torch.nan_to_num(logits, nan=-1e9, posinf=1e9, neginf=-1e9)
            for sid in [MASK_ID, EOS_ID]:
                logits[:, :, sid] = -1e9
        return logits

    with torch.no_grad():
        out = model.diffusion_generate(
            x,
            generation_config=cfg,
            mask_token_id=MASK_ID,
            generation_logits_hook_func=suppress_hook,
        )

    plen = ids.shape[1]
    gen = out[0, plen: plen + max_new_tokens].tolist()
    text = decode(gen)
    unique = len(set(gen))

    print(f"{'=' * 60}")
    print(f"TEST: {label}")
    print(f"  tokens ({unique} unique): {gen[:10]}")
    print(f"  decoded: {text[:120]!r}")
    status = "OK" if unique > 4 and text.strip() else "DEGENERATE"
    print(f"  status: {status}")
    print()
    return status


run(
    "GitHub README example — open generation",
    messages=[{"role": "user", "content": "Please write a poem about the ocean."}],
    max_new_tokens=64,
    steps=512,
    temperature=0.3,
)

run(
    "Simple factual QA — no context",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    max_new_tokens=16,
    steps=256,
    temperature=0.3,
)

run(
    "QA with context — PaRaDe format",
    messages=[
        {
            "role": "user",
            "content": (
                "Context: France is a country in Western Europe. "
                "Its capital is Paris.\n\n"
                "Question: What is the capital of France?\n\nAnswer:"
            ),
        }
    ],
    max_new_tokens=16,
    steps=256,
    temperature=0.3,
)

run(
    "Completion style — no question framing",
    messages=[{"role": "user", "content": "The capital of France is"}],
    max_new_tokens=8,
    steps=256,
    temperature=0.3,
)

print("Done. If TEST 1 is DEGENERATE, Dream cannot generate in this env.")
print("If TEST 1 is OK but others fail, it is a prompt format issue.")
