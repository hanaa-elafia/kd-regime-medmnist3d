"""Etape 1, point 5 - Sonde de resolution : 28^3 vs 64^3.

Un petit CNN 3D generique (PAS un des 3 students) est entraine sur train,
le meilleur checkpoint est choisi sur la validation (Macro-F1).
Le TEST N'EST JAMAIS CHARGE ni utilise ici.

Lancer :
    python scripts/02_resolution_probe.py
Sortie : outputs/step1/resolution_probe.csv + resume dans le terminal
"""
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from medmnist import OrganMNIST3D
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

SEEDS = [0, 1, 2]
SIZES = [28, 64]
EPOCHS = 50
BATCH = 32
LR = 1e-3
N_CLS = 11
dev = torch.device("cuda:0")


def load(split, size):
    ds = OrganMNIST3D(split=split, download=True, size=size)
    x = torch.from_numpy(ds.imgs).float().div(255.0).unsqueeze(1)  # N,1,D,H,W
    y = torch.from_numpy(ds.labels.flatten()).long()
    return x, y


def probe_cnn(ch=(16, 32, 64, 128)):
    layers, cin = [], 1
    for c in ch:
        layers += [nn.Conv3d(cin, c, 3, padding=1), nn.BatchNorm3d(c),
                   nn.ReLU(inplace=True), nn.MaxPool3d(2)]
        cin = c
    return nn.Sequential(*layers, nn.AdaptiveAvgPool3d(1), nn.Flatten(),
                         nn.Linear(cin, N_CLS))


@torch.no_grad()
def predict(model, x):
    model.eval()
    out = []
    for i in range(0, len(x), 64):
        out.append(model(x[i:i + 64].to(dev)).argmax(1).cpu())
    return torch.cat(out).numpy()


rows = []
for size in SIZES:
    xtr, ytr = load("train", size)
    xva, yva = load("val", size)
    mean, std = xtr.mean().item(), xtr.std().item()  # stats du TRAIN seulement
    xtr, xva = (xtr - mean) / std, (xva - mean) / std
    xtr_g, ytr_g = xtr.to(dev), ytr.to(dev)
    yva_np = yva.numpy()

    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = probe_cnn().to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
        best = {"f1": -1, "ep": -1, "acc": 0, "bacc": 0}
        t0 = time.time()
        for ep in range(EPOCHS):
            model.train()
            perm = torch.randperm(len(xtr_g), device=dev)
            for i in range(0, len(perm), BATCH):
                idx = perm[i:i + BATCH]
                if len(idx) < 2:
                    continue
                opt.zero_grad()
                loss = nn.functional.cross_entropy(model(xtr_g[idx]), ytr_g[idx])
                loss.backward()
                opt.step()
            sched.step()
            pred = predict(model, xva)
            f1 = f1_score(yva_np, pred, average="macro")
            if f1 > best["f1"]:
                best = {"f1": f1, "ep": ep + 1,
                        "acc": accuracy_score(yva_np, pred),
                        "bacc": balanced_accuracy_score(yva_np, pred)}
        dt = time.time() - t0
        rows.append({"size": size, "seed": seed, "val_macro_f1": best["f1"],
                     "val_acc": best["acc"], "val_bal_acc": best["bacc"],
                     "best_epoch": best["ep"], "train_time_s": dt})
        print(f"{size}^3 seed={seed} | val F1={best['f1']:.4f} "
              f"acc={best['acc']:.4f} bacc={best['bacc']:.4f} "
              f"best_ep={best['ep']} | {dt:.0f}s", flush=True)
        del model, opt
        torch.cuda.empty_cache()

df = pd.DataFrame(rows)
df.to_csv("outputs/step1/resolution_probe.csv", index=False)
print("\n=== Resume (validation uniquement) ===")
print(df.groupby("size")[["val_macro_f1", "val_acc", "val_bal_acc",
                          "best_epoch", "train_time_s"]]
      .agg(["mean", "std"]).round(4).to_string())
print("\nEnvoie-moi tout le texte ci-dessus.")
