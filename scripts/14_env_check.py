"""
14_env_check.py - Verification d'un nouvel environnement AVANT de lancer B2a.
Validation seulement : aucune donnee de test n'est evaluee.
Usage (depuis KD_FTS_FAS_DFS) :
    python scripts/14_env_check.py 2>&1 | tee outputs/b2a_env_check.txt
Code de sortie 1 si une verification echoue : dans ce cas NE PAS lancer B2a.
"""
import hashlib
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import medmnist
from medmnist import INFO

DATASETS = {  # tailles train/val/test, moyenne et ecart-type du train (valeurs de la Phase B)
    "organmnist3d": ((971, 161, 610), 0.5009, 0.2802),
    "adrenalmnist3d": ((1188, 98, 298), 0.0158, 0.1247),
    "fracturemnist3d": ((1027, 103, 240), 0.0229, 0.1104),
}
TEACHERS = ["res_w16_b1", "res_w32_b1", "res_w32_b2"]
fails = []


def check(ok, msg):
    print(("OK      " if ok else "ECHEC   ") + msg)
    if not ok:
        fails.append(msg)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class Block(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv3d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm3d(cout)
        self.c2 = nn.Conv3d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm3d(cout)
        self.sc = None
        if stride != 1 or cin != cout:
            self.sc = nn.Sequential(nn.Conv3d(cin, cout, 1, stride, bias=False), nn.BatchNorm3d(cout))

    def forward(self, x):
        o = F.relu(self.b1(self.c1(x)))
        o = self.b2(self.c2(o))
        return F.relu(o + (x if self.sc is None else self.sc(x)))


def resnet3d(widths, blocks, n_cls):
    layers = [nn.Conv3d(1, widths[0], 3, 1, 1, bias=False), nn.BatchNorm3d(widths[0]), nn.ReLU(inplace=True)]
    cin = widths[0]
    for i, (w, nb) in enumerate(zip(widths, blocks)):
        for j in range(nb):
            layers.append(Block(cin, w, 2 if (j == 0 and i > 0) else 1))
            cin = w
    return nn.Sequential(*layers, nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Linear(cin, n_cls))


MODELS = {
    "res_w16_b1": ((16, 32, 64, 128), (1, 1, 1, 1)),
    "res_w32_b1": ((32, 64, 128, 256), (1, 1, 1, 1)),
    "res_w32_b2": ((32, 64, 128, 256), (2, 2, 2, 2)),
}

print("== Versions ==")
import scipy, sklearn
print(f"python {sys.version.split()[0]} | torch {torch.__version__} | medmnist {medmnist.__version__} | "
      f"numpy {np.__version__} | pandas {pd.__version__} | scipy {scipy.__version__} | sklearn {sklearn.__version__}")
print(f"cuda disponible : {torch.cuda.is_available()} | nb GPU : {torch.cuda.device_count()}")
print("(Compare avec l'ancien environnement : conda list | grep -E 'torch|medmnist|numpy|pandas|scipy|scikit')")

print("\n== Scripts ==")
for s in ["09_phase_b.py", "13_b2a_analysis.py", "10_analysis.py"]:
    p = f"scripts/{s}"
    check(os.path.exists(p), f"{p} present" + (f" (sha256 {sha(p)[:16]}...)" if os.path.exists(p) else ""))
if os.path.exists("scripts/09_phase_b.py"):
    check("b2a" in open("scripts/09_phase_b.py").read(), "09_phase_b.py contient le plan b2a")

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
for ds, (sizes, mu, sd) in DATASETS.items():
    print(f"\n== {ds} ==")
    info = INFO[ds]
    DC = getattr(medmnist, info["python_class"])
    n_cls = len(info["label"])

    def load(split):
        m = DC(split=split, download=True, size=28)
        x = torch.from_numpy(m.imgs).float().div(255.0)
        x = x.unsqueeze(1) if x.ndim == 4 else x
        return x, torch.from_numpy(m.labels.flatten()).long()

    (xtr, _), (xva, yva) = load("train"), load("val")
    xte_n = len(DC(split="test", download=True, size=28))  # taille seulement, pas d'evaluation
    check((len(xtr), len(xva), xte_n) == sizes, f"tailles {len(xtr)}/{len(xva)}/{xte_n} attendues {sizes}")
    mean, std = xtr.mean().item(), xtr.std().item()
    check(abs(mean - mu) < 5e-4 and abs(std - sd) < 5e-4, f"moyenne/ecart-type train {mean:.4f}/{std:.4f} attendus {mu}/{sd}")
    xva = (xva - mean) / std
    d = f"outputs/phaseB_{ds}"
    tp = f"{d}/teachers_results.csv"
    if not os.path.exists(tp):
        check(False, f"{tp} absent (copier outputs/phaseB_* de l'ancien environnement)")
        continue
    tr = pd.read_csv(tp).set_index("teacher")
    for t in TEACHERS:
        pt = f"{d}/teacher_{t}.pt"
        if not os.path.exists(pt):
            check(False, f"{pt} absent")
            continue
        w, b = MODELS[t]
        m = resnet3d(w, b, n_cls).to(dev)
        m.load_state_dict(torch.load(pt, map_location=dev))
        m.eval()
        with torch.no_grad():
            lg = torch.cat([m(xva[i:i + 64].to(dev)).cpu() for i in range(0, len(xva), 64)])
        v = F.cross_entropy(lg, yva).item()
        rec = float(tr.loc[t, "val_nll"])
        check(abs(v - rec) <= 1e-4, f"{t} : NLL validation recalculee {v:.5f} vs enregistree {rec:.5f} (sha {sha(pt)[:12]})")

print()
if fails:
    print(f"RESULTAT : {len(fails)} ECHEC(S) -> NE PAS LANCER B2a dans cet environnement.")
    sys.exit(1)
print("RESULTAT : tout est OK -> B2a peut etre lance (apres le --smoke).")
