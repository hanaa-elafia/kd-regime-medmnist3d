#!/bin/bash
# Phase B2a - GPU 0 : Student seul (3 students) + Vanilla avec res_w16_b1 et res_w32_b1.
# A lancer depuis ~/Documents/Hanaa2/KD_FTS_FAS_DFS, environnement kd-medmnist actif,
# APRES avoir date l'addendum et teste le plan b2a avec --smoke.
#   nohup bash scripts/run_b2a_gpu0.sh > outputs/log_b2a_gpu0.txt 2>&1 &
# Reprise : si le script s'arrete, relancer la meme commande (les runs faits sont sautes).
set -e
export CUDA_VISIBLE_DEVICES=0
for DS in organmnist3d adrenalmnist3d fracturemnist3d; do
  python scripts/09_phase_b.py --stage run --plan b2a --dataset $DS --teachers res_w16_b1,res_w32_b1 --tag _gpu0
done
echo "B2a GPU 0 : tout est termine."
