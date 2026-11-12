#!/bin/bash
# Phase B - GPU 1 : teacher T3 avec S1 et S2, puis le bloc pivot (res_w32_b2 x S1, seeds 5 a 14).
# A lancer depuis ~/Documents/Hanaa2/KD_FTS_FAS_DFS, avec l'environnement kd-medmnist actif,
# APRES l'etape des teachers (les 9 fichiers teacher_*.pt doivent exister).
#   nohup bash scripts/run_phaseB_gpu1.sh > outputs/log_phaseB_gpu1.txt 2>&1 &
# Reprise : si le script s'arrete, relance la meme commande (les runs deja faits sont sautes).
set -e
export CUDA_VISIBLE_DEVICES=1
for DS in organmnist3d adrenalmnist3d fracturemnist3d; do
  python scripts/09_phase_b.py --stage run --plan primary --dataset $DS --teachers res_w32_b2 --students S1,S2 --tag _gpu1
  python scripts/09_phase_b.py --stage run --plan pivot --dataset $DS --tag _gpu1
done
echo "GPU 1 : tout est termine."
