#!/bin/bash
# setup_env.sh
# Run this once on your H200 machine before anything else.
# Usage: bash setup_env.sh [--model_path /path/to/LLaDA-8B-Instruct]

set -e

MODEL_PATH=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path) MODEL_PATH="$2"; shift 2 ;;
        *) shift ;;
    esac
done

echo "============================================"
echo " DLLM Hallucination - Environment Setup"
echo "============================================"

# 1. Python version check
echo ""
echo "[1/6] Checking Python version..."
if python3 -c "import sys; assert sys.version_info >= (3,10)" 2>/dev/null; then
    echo "  [OK] Python $(python3 --version | awk '{print $2}')"
else
    echo "  [FAIL] Need Python >= 3.10"
    exit 1
fi

# 2. CUDA check
echo ""
echo "[2/6] Checking CUDA / GPU..."
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader \
        | sed 's/^/  /'
    echo "  [OK] GPU detected"
else
    echo "  [FAIL] nvidia-smi not found"
    exit 1
fi

# 3. Virtual environment
echo ""
echo "[3/6] Creating virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "  [OK] Created .venv"
else
    echo "  [OK] .venv already exists"
fi
source .venv/bin/activate

# 4. Install dependencies
echo ""
echo "[4/6] Installing dependencies..."
python3 -m pip install --upgrade pip --quiet
python3 -m pip install -r requirements.txt --quiet
python3 -m pip install -e ".[dev]" --quiet
echo "  [OK] Dependencies installed"

# 5. Model path check
echo ""
echo "[5/6] Checking local model path..."
if [ -n "$MODEL_PATH" ]; then
    if [ -d "$MODEL_PATH" ]; then
        config="$MODEL_PATH/config.json"
        if [ -f "$config" ]; then
            model_type=$(python3 -c "import json; d=json.load(open('$config')); print(d.get('_name_or_path', d.get('model_type','?')))" 2>/dev/null || echo "?")
            echo "  [OK] Model found at: $MODEL_PATH"
            echo "  [..] model type: $model_type"
            if echo "$MODEL_PATH" | grep -qi "instruct"; then
                echo "  [OK] Confirmed: using Instruct variant"
            else
                echo "  [WARN] Path does not contain 'instruct' - make sure this is LLaDA-8B-Instruct"
                echo "         Base model will not work correctly for RAG-style prompts"
            fi
        else
            echo "  [FAIL] config.json not found in $MODEL_PATH"
            exit 1
        fi
    else
        echo "  [FAIL] Path does not exist: $MODEL_PATH"
        exit 1
    fi
else
    echo "  [..] No --model_path provided; runtime will use a configured local alias/path"
fi

# 6. Quick torch + CUDA smoke test
echo ""
echo "[6/6] PyTorch + CUDA smoke test..."
python3 - <<'EOF'
import torch

print(f"  torch : {torch.__version__}")
print(f"  CUDA  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        vram = p.total_memory / 1024**3
        print(f"  GPU {i}: {p.name}  ({vram:.0f} GB VRAM)")
        if vram < 40:
            print(f"  [WARN] GPU {i} has < 40GB - 4 paths at gen_len=64 may OOM")
        else:
            print(f"  [OK] GPU {i} has enough VRAM for 4 parallel paths")
    x = torch.ones(4, 4, dtype=torch.bfloat16, device="cuda")
    _ = x @ x
    print("  [OK] bfloat16 matmul OK")
EOF

echo ""
echo "============================================"
echo " Setup complete. Next steps:"
echo "   source .venv/bin/activate"
echo "   python sanitycheck.py --model_path /your/local/path \\"
echo "                         --data_path  /your/local/ragtruth"
echo "   python run_experiment.py --model_id /your/local/path"
echo "============================================"
