"""Etape 1 - Balayage de capacite a 28^3 (validation uniquement).

Objectif : trouver a partir de quelle taille le student seul n'est plus
sature, pour guider le choix de S1 < S2 < S3 (pas encore le choix final).
Le checkpoint est choisi sur la NLL de validation (pas sur le F1).
Le TEST n'est jamais charge.

Lancer :
    python scripts/03_capacity_sweep.py
Sortie : outputs/step1/capacity_sweep.csv + resume dans le terminal
"""
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from medmnist import OrganMNIST3D
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

SEEDS = [0, 1, 2]
SIZE = 28
WIDTHS = [(2, 4, 8, 16), (4, 8, 16, 32), (8, 16, 32, 64), (16, 32, 64, 128)]
EPOCHS = 50
BATCH = 32
LR = 1e-3
N_CLS = 11
dev = torch.device("cuda:0")


def load(split):
    ds = OrganMNIST3D(split=split, download=True, size=SIZE)
    x = torch.from_numpy(ds.imgs).float().div(255.0).unsqueeze(1)
    y = torch.from_numpy(ds.labels.flatten()).long()
    return x, y


def make_cnn(ch):
    layers, cin = [], 1
    for c in ch:
        layers += [nn.Conv3d(cin, c, 3, padding=1), nn.BatchNorm3d(c),
                   nn.ReLU(inplace=True), nn.MaxPool3d(2)]
        cin = c
    return nn.Sequential(*layers, nn.AdaptiveAvgPool3d(1), nn.Flatten(),
                         nn.Linear(cin, N_CLS))


@torch.no_grad()
def val_outputs(model, x):
    model.eval()
    logits = torch.cat([model(x[i:i + 64].to(dev)).cpu()
                        for i in range(0, len(x), 64)])
    return logits


def ece_score(probs, y, n_bins=15):
    conf, pred = probs.max(1)
    correct = (pred == y).float()
    edges = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.float().mean().item() * abs(
                correct[m].mean().item() - conf[m].mean().item())
    return ece


xtr, ytr = load("train")
xva, yva = load("val")
mean, std = xtr.mean().item(), xtr.std().item()  # stats du TRAIN seulement
xtr, xva = (xtr - mean) / std, (xva - mean) / std
xtr_g, ytr_g = xtr.to(dev), ytr.to(dev)
yva_np = yva.numpy()

rows = []
for ch in WIDTHS:
    n_params = sum(p.numel() for p in make_cnn(ch).parameters())
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = make_cnn(ch).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
        best = None
        t0 = time.time()
        for ep in range(EPOCHS):
            model.train()
            perm = torch.randperm(len(xtr_g), device=dev)
            for i in range(0, len(perm), BATCH):
                idx = perm[i:i + BATCH]
                if len(idx) < 2:
                    continue
                opt.zero_grad()
                F.cross_entropy(model(xtr_g[idx]), ytr_g[idx]).backward()
                opt.step()
            sched.step()
            logits = val_outputs(model, xva)
            nll = F.cross_entropy(logits, yva).item()
            if best is None or nll < best["nll"]:
                probs = logits.softmax(1)
                pred = probs.argmax(1).numpy()
                best = {"nll": nll, "ep": ep + 1,
                        "f1": f1_score(yva_np, pred, average="macro"),
                        "acc": accuracy_score(yva_np, pred),
                        "bacc": balanced_accuracy_score(yva_np, pred),
                        "ece": ece_score(probs, yva)}
        dt = time.time() - t0
        rows.append({"widths": str(ch), "params": n_params, "seed": seed,
                     "val_nll": best["nll"], "val_macro_f1": best["f1"],
                     "val_acc": best["acc"], "val_bal_acc": best["bacc"],
                     "val_ece": best["ece"], "best_epoch": best["ep"],
                     "train_time_s": dt})
        print(f"{str(ch):<18} params={n_params:>7} seed={seed} | "
              f"NLL={best['nll']:.4f} F1={best['f1']:.4f} "
              f"ECE={best['ece']:.4f} best_ep={best['ep']} | {dt:.0f}s",
              flush=True)
        del model, opt
        torch.cuda.empty_cache()

df = pd.DataFrame(rows)
df.to_csv("outputs/step1/capacity_sweep.csv", index=False)
print("\n=== Resume (validation uniquement, checkpoint choisi sur NLL) ===")
summ = (df.groupby(["widths", "params"])
        [["val_nll", "val_macro_f1", "val_bal_acc", "val_ece", "best_epoch"]]
        .agg(["mean", "std"]).round(4))
print(summ.to_string())
print("\nEnvoie-moi tout le texte ci-dessus.")
