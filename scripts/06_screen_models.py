"""Criblage generique : students S1-S3 + candidats teachers sur un dataset.

Meme protocole pour tous les datasets : 80 epochs, AdamW lr 1e-3, batch 32,
cosine, sans augmentation, normalisation avec les stats du TRAIN,
checkpoint choisi sur la NLL de validation. Le TEST n'est jamais charge.

Exemples :
    CUDA_VISIBLE_DEVICES=0 python scripts/06_screen_models.py --dataset fracturemnist3d \
        --models S1,S2,S3,res_w32_b2 --tag _gpu0
    CUDA_VISIBLE_DEVICES=1 python scripts/06_screen_models.py --dataset fracturemnist3d \
        --models res_w32_b1,res_w16_b1,plain_w32 --tag _gpu1

Sortie : outputs/screen_<dataset>/results<tag>.csv (ecrit apres chaque run)
"""
import argparse
import os
import time

import medmnist
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from medmnist import INFO
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             roc_auc_score)
from torch.utils.flop_counter import FlopCounterMode

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
parser.add_argument("--models", default="all",
                    help="liste separee par des virgules, ou 'all'")
parser.add_argument("--seeds", type=int, default=5)
parser.add_argument("--epochs", type=int, default=80)
parser.add_argument("--size", type=int, default=28)
parser.add_argument("--tag", default="")
parser.add_argument("--aug", action="store_true",
                    help="augmentation legere sans flips (translation, intensite, bruit)")
parser.add_argument("--wd", type=float, default=1e-4, help="weight decay")
args = parser.parse_args()

FLAG = args.dataset.lower()
info = INFO[FLAG]
DataClass = getattr(medmnist, info["python_class"])
N_CLS = len(info["label"])
BATCH, LR = 32, 1e-3
dev = torch.device("cuda:0")
OUT = f"outputs/screen_{FLAG}" + ("_aug" if args.aug else "")
os.makedirs(OUT, exist_ok=True)


# ----------------------------------------------------------------- modeles
def plain_cnn(ch):
    layers, cin = [], 1
    for c in ch:
        layers += [nn.Conv3d(cin, c, 3, padding=1), nn.BatchNorm3d(c),
                   nn.ReLU(inplace=True), nn.MaxPool3d(2)]
        cin = c
    return nn.Sequential(*layers, nn.AdaptiveAvgPool3d(1), nn.Flatten(),
                         nn.Linear(cin, N_CLS))


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv3d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm3d(cout)
        self.c2 = nn.Conv3d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm3d(cout)
        self.sc = None
        if stride != 1 or cin != cout:
            self.sc = nn.Sequential(nn.Conv3d(cin, cout, 1, stride, bias=False),
                                    nn.BatchNorm3d(cout))

    def forward(self, x):
        out = F.relu(self.b1(self.c1(x)))
        out = self.b2(self.c2(out))
        return F.relu(out + (x if self.sc is None else self.sc(x)))


def resnet3d(widths, blocks):
    layers = [nn.Conv3d(1, widths[0], 3, 1, 1, bias=False),
              nn.BatchNorm3d(widths[0]), nn.ReLU(inplace=True)]
    cin = widths[0]
    for i, (w, nb) in enumerate(zip(widths, blocks)):
        for j in range(nb):
            layers.append(BasicBlock(cin, w, 2 if (j == 0 and i > 0) else 1))
            cin = w
    return nn.Sequential(*layers, nn.AdaptiveAvgPool3d(1), nn.Flatten(),
                         nn.Linear(cin, N_CLS))


MODELS = {
    "S1": ("student", lambda: plain_cnn((2, 4, 8, 16))),
    "S2": ("student", lambda: plain_cnn((4, 8, 16, 32))),
    "S3": ("student", lambda: plain_cnn((8, 16, 32, 64))),
    "res_w16_b1": ("teacher", lambda: resnet3d((16, 32, 64, 128), (1, 1, 1, 1))),
    "plain_w32": ("teacher", lambda: plain_cnn((32, 64, 128, 256))),
    "res_w32_b1": ("teacher", lambda: resnet3d((32, 64, 128, 256), (1, 1, 1, 1))),
    "res_w32_b2": ("teacher", lambda: resnet3d((32, 64, 128, 256), (2, 2, 2, 2))),
}
names = list(MODELS) if args.models == "all" else args.models.split(",")
for n in names:
    assert n in MODELS, f"modele inconnu : {n} (choix : {list(MODELS)})"


def complexity(make):
    m = make().eval()
    n_params = sum(p.numel() for p in m.parameters())
    with FlopCounterMode(display=False) as fc:
        m(torch.zeros(1, 1, args.size, args.size, args.size))
    return n_params, fc.get_total_flops() / 1e6


# ------------------------------------------------------------------ donnees
def load(split):
    ds = DataClass(split=split, download=True, size=args.size)
    x = torch.from_numpy(ds.imgs).float().div(255.0)
    x = x.unsqueeze(1) if x.ndim == 4 else x
    return x, torch.from_numpy(ds.labels.flatten()).long()


@torch.no_grad()
def val_logits(model, x):
    model.eval()
    return torch.cat([model(x[i:i + 64].to(dev)).cpu()
                      for i in range(0, len(x), 64)])


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


def augment(x):
    """Translation +-2 voxels, echelle/decalage d'intensite, bruit. PAS de flips
    (les etiquettes gauche/droite dependent de la lateralite)."""
    B, _, D, H, W = x.shape
    xp = F.pad(x, (2,) * 6, mode="replicate")
    out = torch.empty_like(x)
    sh = torch.randint(0, 5, (B, 3)).tolist()
    for i, (a, b, c) in enumerate(sh):
        out[i] = xp[i, :, a:a + D, b:b + H, c:c + W]
    scale = 1 + 0.1 * (torch.rand(B, 1, 1, 1, 1, device=x.device) * 2 - 1)
    shift = 0.1 * torch.randn(B, 1, 1, 1, 1, device=x.device)
    return out * scale + shift + 0.05 * torch.randn_like(out)


def auc_score(probs, y):
    try:
        if N_CLS == 2:
            return roc_auc_score(y, probs[:, 1])
        return roc_auc_score(y, probs, multi_class="ovr", average="macro",
                             labels=list(range(N_CLS)))
    except ValueError:
        return float("nan")


print(f"Dataset {FLAG} | {N_CLS} classes | {args.size}^3 | "
      f"{args.seeds} seeds | {args.epochs} epochs | aug={args.aug} wd={args.wd}")
comp = {}
for n in names:
    comp[n] = complexity(MODELS[n][1])
    print(f"{n:<12} {MODELS[n][0]:<8} params={comp[n][0]:>9,} "
          f"{comp[n][1]:>9.1f} MFLOPs")
print()

xtr, ytr = load("train")
xva, yva = load("val")
mean, std = xtr.mean().item(), xtr.std().item()  # stats du TRAIN seulement
xtr, xva = (xtr - mean) / std, (xva - mean) / std
xtr_g, ytr_g = xtr.to(dev), ytr.to(dev)
yva_np = yva.numpy()
print(f"train N={len(ytr)} val N={len(yva)} | mean={mean:.4f} std={std:.4f}\n")

rows, csv_path = [], f"{OUT}/results{args.tag}.csv"
for name in names:
    role, make = MODELS[name]
    for seed in range(args.seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = make().to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=args.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
        best, t0 = None, time.time()
        for ep in range(args.epochs):
            model.train()
            perm = torch.randperm(len(xtr_g), device=dev)
            for i in range(0, len(perm), BATCH):
                idx = perm[i:i + BATCH]
                if len(idx) < 2:
                    continue
                opt.zero_grad()
                xb = augment(xtr_g[idx]) if args.aug else xtr_g[idx]
                F.cross_entropy(model(xb), ytr_g[idx]).backward()
                opt.step()
            sched.step()
            logits = val_logits(model, xva)
            nll = F.cross_entropy(logits, yva).item()
            if best is None or nll < best["nll"]:
                probs = logits.softmax(1)
                pred = probs.argmax(1).numpy()
                best = {"nll": nll, "ep": ep + 1,
                        "f1": f1_score(yva_np, pred, average="macro"),
                        "bacc": balanced_accuracy_score(yva_np, pred),
                        "acc": accuracy_score(yva_np, pred),
                        "ece": ece_score(probs, yva),
                        "auc": auc_score(probs.numpy(), yva_np)}
        dt = time.time() - t0
        rows.append({"dataset": FLAG, "model": name, "role": role,
                     "params": comp[name][0], "mflops": comp[name][1],
                     "seed": seed, "val_nll": best["nll"],
                     "val_macro_f1": best["f1"], "val_bal_acc": best["bacc"],
                     "val_acc": best["acc"], "val_ece": best["ece"],
                     "val_auc": best["auc"], "best_epoch": best["ep"],
                     "train_time_s": dt})
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"{name:<12} seed={seed} | NLL={best['nll']:.4f} "
              f"F1={best['f1']:.4f} AUC={best['auc']:.4f} "
              f"ECE={best['ece']:.4f} best_ep={best['ep']} | {dt:.0f}s",
              flush=True)
        del model, opt
        torch.cuda.empty_cache()

df = pd.DataFrame(rows)
print("\n=== Resume (validation uniquement, checkpoint choisi sur NLL) ===")
summ = (df.groupby(["model", "params"])
        [["val_nll", "val_macro_f1", "val_bal_acc", "val_auc", "val_ece",
          "best_epoch"]].agg(["mean", "std"]).round(4))
print(summ.to_string())
print(f"\nFichier : {csv_path}")
print("Envoie-moi tout le texte ci-dessus.")
