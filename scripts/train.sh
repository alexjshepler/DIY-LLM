#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/small.yaml}

# 1) prepare
python scripts/prepare_data.py --config "$CONFIG"

# 2) train
python train.py --config "$CONFIG"

# 3) sample
python scripts/sample.py --config "$CONFIG" --prompt "Once upon a time," --max_new_tokens 120 --temperature 0.9 --top_k 200
