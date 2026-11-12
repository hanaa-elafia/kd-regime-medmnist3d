"""Etape 1, point 8 - Criblage des candidats teachers a 28^3.

Objectif : mesurer des candidats teachers (familles et tailles differentes)
et les comparer aux students S1/S2/S3 (params, FLOPs) et en qualite.
Checkpoint choisi sur la NLL de validation. Le TEST n'est jamais charge.

Lancer :
    python scripts/04_teacher_candidates.py
Sortie : outputs/step1/teacher_candidates.csv + resume dans le terminal
"""
import argparse
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from medmnist import OrganMNIST3D
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch.utils.flop_counter import FlopCounterMode

SEEDS = [0, 1, 2]
SIZE = 28
EPOCHS = 80          # a fixer definitivement dans le protocole
BATCH = 32
LR = 1e-3
N_CLS = 11
dev = torch.device("cuda:0")


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
            stride = 2 if (j == 0 and i > 0) else 1
            layers.append(BasicBlock(cin, w, stride))
            cin = w
    return nn.Sequential(*layers, nn.AdaptiveAvgPool3d(1), nn.Flatten(),
                         nn.Linear(cin, N_CLS))


CANDIDATES = {
    "plain_w32":   lambda: plain_cnn((32, 64, 128, 256)),
    "res_w16_b1":  lambda: resnet3d((16, 32, 64, 128), (1, 1, 1, 1)),
    "res_w32_b1":  lambda: resnet3d((32, 64, 128, 256), (1, 1, 1, 1)),
    "res_w32_b2":  lambda: resnet3d((32, 64, 128, 256), (2, 2, 2, 2)),
}
STUDENTS = {  # reference : params/FLOPs seulement (pas entraines ici)
    "S1_(2,4,8,16)":   lambda: plain_cnn((2, 4, 8, 16)),
    "S2_(4,8,16,32)":  lambda: plain_cnn((4, 8, 16, 32)),
    "S3_(8,16,32,64)": lambda: plain_cnn((8, 16, 32, 64)),
}


parser = argparse.ArgumentParser()
parser.add_argument("--only", default="",
                    help="candidats a entrainer, separes par des virgules")
parser.add_argument("--tag", default="", help="suffixe du fichier de sortie")
args = parser.parse_args()
TO_TRAIN = ({k: v for k, v in CANDIDATES.items() if k in args.only.split(",")}
            if args.only else CANDIDATES)


def complexity(make):
    m = make().eval()
    n_params = sum(p.numel() for p in m.parameters())
    with FlopCounterMode(display=False) as fc:
        m(torch.zeros(1, 1, SIZE, SIZE, SIZE))
    return n_params, fc.get_total_flops() / 1e6  # MFLOPs


# ------------------------------------------------------------------ donnees
def load(split):
    ds = OrganMNIST3D(split=split, download=True, size=SIZE)
    x = torch.from_numpy(ds.imgs).float().div(255.0).unsqueeze(1)
    y = torch.from_numpy(ds.labels.flatten()).long()
    return x, y


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


print("=== Complexite (forward, 1 volume 28^3) ===")
comp = {}
for name, mk in {**STUDENTS, **CANDIDATES}.items():
    p, mf = complexity(mk)
    comp[name] = (p, mf)
    print(f"{name:<18} params={p:>9,}  {mf:>10.1f} MFLOPs")
print()

xtr, ytr = load("train")
xva, yva = load("val")
mean, std = xtr.mean().item(), xtr.std().item()   # stats du TRAIN seulement
xtr, xva = (xtr - mean) / std, (xva - mean) / std
xtr_g, ytr_g = xtr.to(dev), ytr.to(dev)
yva_np = yva.numpy()

rows = []
for name, mk in TO_TRAIN.items():
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = mk().to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
        best, t0 = None, time.time()
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
            logits = val_logits(model, xva)
            nll = F.cross_entropy(logits, yva).item()
            if best is None or nll < best["nll"]:
                probs = logits.softmax(1)
                pred = probs.argmax(1).numpy()
                best = {"nll": nll, "ep": ep + 1,
                        "f1": f1_score(yva_np, pred, average="macro"),
                        "bacc": balanced_accuracy_score(yva_np, pred),
                        "acc": accuracy_score(yva_np, pred),
                        "ece": ece_score(probs, yva)}
        dt = time.time() - t0
        rows.append({"model": name, "params": comp[name][0],
                     "mflops": comp[name][1], "seed": seed,
                     "val_nll": best["nll"], "val_macro_f1": best["f1"],
                     "val_bal_acc": best["bacc"], "val_acc": best["acc"],
                     "val_ece": best["ece"], "best_epoch": best["ep"],
                     "train_time_s": dt})
        print(f"{name:<12} seed={seed} | NLL={best['nll']:.4f} "
              f"F1={best['f1']:.4f} ECE={best['ece']:.4f} "
              f"best_ep={best['ep']} | {dt:.0f}s", flush=True)
        del model, opt
        torch.cuda.empty_cache()

df = pd.DataFrame(rows)
df.to_csv(f"outputs/step1/teacher_candidates{args.tag}.csv", index=False)
print("\n=== Resume (validation uniquement, checkpoint choisi sur NLL) ===")
summ = (df.groupby(["model", "params"])
        [["val_nll", "val_macro_f1", "val_bal_acc", "val_ece",
          "best_epoch", "train_time_s"]]
        .agg(["mean", "std"]).round(4))
print(summ.to_string())
print("\nEnvoie-moi tout le texte ci-dessus (complexite incluse).")
