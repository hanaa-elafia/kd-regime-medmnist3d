#!/bin/bash
# Phase B2a - GPU 1 : Vanilla avec res_w32_b2 (sans les Student seuls, lances sur le GPU 0).
#   nohup bash scripts/run_b2a_gpu1.sh > outputs/log_b2a_gpu1.txt 2>&1 &
set -e
export CUDA_VISIBLE_DEVICES=1
for DS in organmnist3d adrenalmnist3d fracturemnist3d; do
  python scripts/09_phase_b.py --stage run --plan b2a --dataset $DS --teachers res_w32_b2 --no_students --tag _gpu1
done
echo "B2a GPU 1 : tout est termine."
