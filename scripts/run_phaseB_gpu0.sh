#!/bin/bash
# Phase B - GPU 0 : Student seul, teachers T1 et T2 (tous les students), puis T3 x S3.
# A lancer depuis ~/Documents/Hanaa2/KD_FTS_FAS_DFS, avec l'environnement kd-medmnist actif,
# APRES l'etape des teachers (les 9 fichiers teacher_*.pt doivent exister).
#   nohup bash scripts/run_phaseB_gpu0.sh > outputs/log_phaseB_gpu0.txt 2>&1 &
# Reprise : si le script s'arrete, relance la meme commande (les runs deja faits sont sautes).
set -e
export CUDA_VISIBLE_DEVICES=0
for DS in organmnist3d adrenalmnist3d fracturemnist3d; do
  python scripts/09_phase_b.py --stage run --plan students --dataset $DS --tag _gpu0
  python scripts/09_phase_b.py --stage run --plan primary --dataset $DS --teachers res_w16_b1,res_w32_b1 --tag _gpu0
  python scripts/09_phase_b.py --stage run --plan primary --dataset $DS --teachers res_w32_b2 --students S3 --tag _gpu0
done
echo "GPU 0 : tout est termine."
