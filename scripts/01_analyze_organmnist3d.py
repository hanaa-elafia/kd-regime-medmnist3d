"""Etape 1 - Analyse complete de OrganMNIST3D.

Couvre : telechargement, splits, classes/desequilibre, doublons (fuite),
visualisation, comparaison 28^3 vs 64^3, cout GPU.

Installation (dans l'env kd-medmnist) :
    pip install medmnist matplotlib numpy torch

Lancer :
    python scripts/01_analyze_organmnist3d.py
Sorties : dossier outputs/step1/ (report.json + images .png)
"""
import hashlib
import json
import os
import time

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from medmnist import INFO, OrganMNIST3D

OUT = "outputs/step1"
os.makedirs(OUT, exist_ok=True)
info = INFO["organmnist3d"]
class_names = {int(k): v for k, v in info["label"].items()}
n_cls = len(class_names)
report = {"classes": class_names, "resolutions": {}}
print("Classes :", class_names)
print("Task :", info["task"], "| channels :", info["n_channels"])


def md5(v):
    return hashlib.md5(v.tobytes()).hexdigest()


# ---------------------------------------------------------------- donnees
for size in (28, 64):
    print(f"\n{'=' * 14} Resolution {size}^3 {'=' * 14}")
    r = {"splits": {}}
    data, hashes = {}, {}
    for split in ("train", "val", "test"):
        ds = OrganMNIST3D(split=split, download=True, size=size)
        x, y = ds.imgs, ds.labels.flatten()
        data[split] = (x, y)
        hashes[split] = [md5(v) for v in x]
        counts = np.bincount(y, minlength=n_cls)
        nz = counts[counts > 0]
        ir = float(nz.max() / nz.min())
        r["splits"][split] = {
            "N": int(len(y)),
            "shape": list(x.shape),
            "dtype": str(x.dtype),
            "min": int(x.min()),
            "max": int(x.max()),
            "counts": counts.tolist(),
            "IR": ir,
            "missing_classes": np.where(counts == 0)[0].tolist(),
        }
        print(f"\n[{split}] N={len(y)} shape={x.shape} dtype={x.dtype} "
              f"range=[{x.min()},{x.max()}]")
        print("  par classe :", counts.tolist())
        print(f"  proportions : {np.round(counts / counts.sum(), 3).tolist()}")
        print(f"  IR = Nmax/Nmin = {ir:.2f}")
        if len(nz) < n_cls:
            print("  ATTENTION classes absentes :", r["splits"][split]["missing_classes"])

    # fuite : volumes identiques entre splits / dans le train
    r["leakage"] = {}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        k = len(set(hashes[a]) & set(hashes[b]))
        r["leakage"][f"{a}/{b}"] = k
        print(f"volumes identiques {a}/{b} : {k}")
    dup_train = len(hashes["train"]) - len(set(hashes["train"]))
    r["leakage"]["duplicates_in_train"] = dup_train
    print(f"doublons internes train : {dup_train}")
    print("NB : l'absence de doublons exacts ne prouve pas l'absence de fuite "
          "patient-level (non verifiable depuis MedMNIST).")

    # statistiques d'intensite (utile pour la normalisation)
    xt = data["train"][0].astype(np.float32) / 255.0
    r["intensity_train"] = {"mean": float(xt.mean()), "std": float(xt.std())}
    print(f"intensite train (0-1) : mean={xt.mean():.4f} std={xt.std():.4f}")

    # visualisation : 3 coupes (axial/coronal/sagittal) pour 1 volume par classe
    xtr, ytr = data["train"]
    fig, axes = plt.subplots(3, n_cls, figsize=(1.8 * n_cls, 5.6))
    for c in range(n_cls):
        idx = np.where(ytr == c)[0]
        for row in range(3):
            ax = axes[row, c]
            ax.axis("off")
            if len(idx) == 0:
                continue
            vol = xtr[idx[0]]
            vol = vol[0] if vol.ndim == 4 else vol
            m = vol.shape[0] // 2
            sl = (vol[m], vol[:, m], vol[:, :, m])[row]
            ax.imshow(sl, cmap="gray")
            if row == 0:
                ax.set_title(f"{c}\n{class_names[c][:10]}", fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{OUT}/examples_{size}.png", dpi=130)
    plt.close()

    # histogramme des classes par split
    fig, ax = plt.subplots(figsize=(8, 3))
    w = 0.27
    for i, s in enumerate(("train", "val", "test")):
        c = np.array(r["splits"][s]["counts"])
        ax.bar(np.arange(n_cls) + (i - 1) * w, c / c.sum(), w, label=s)
    ax.set_xlabel("classe")
    ax.set_ylabel("proportion")
    ax.set_xticks(range(n_cls))
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT}/class_dist_{size}.png", dpi=130)
    plt.close()
    report["resolutions"][str(size)] = r

# ---------------------------------------------------------------- GPU
print(f"\n{'=' * 14} Cout GPU {'=' * 14}")
try:
    import torch
    import torch.nn as nn

    if not torch.cuda.is_available():
        print("CUDA indisponible : test GPU saute.")
        report["gpu"] = "cuda_unavailable"
    else:
        n_gpu = torch.cuda.device_count()
        report["gpu"] = {}
        for g in range(n_gpu):
            p = torch.cuda.get_device_properties(g)
            print(f"GPU {g} : {p.name}, {p.total_memory / 1e9:.1f} Go")

        def tiny_cnn(ch=(16, 32, 64, 128)):
            layers, cin = [], 1
            for c in ch:
                layers += [nn.Conv3d(cin, c, 3, padding=1), nn.BatchNorm3d(c),
                           nn.ReLU(), nn.MaxPool3d(2)]
                cin = c
            return nn.Sequential(*layers, nn.AdaptiveAvgPool3d(1), nn.Flatten(),
                                 nn.Linear(cin, n_cls))

        dev = torch.device("cuda:0")
        for size in (28, 64):
            for bs in (16, 32, 64):
                model = tiny_cnn().to(dev)
                opt = torch.optim.Adam(model.parameters(), 1e-3)
                x = torch.randn(bs, 1, size, size, size, device=dev)
                y = torch.randint(0, n_cls, (bs,), device=dev)
                torch.cuda.reset_peak_memory_stats(dev)
                try:
                    for _ in range(3):  # warm-up
                        opt.zero_grad()
                        nn.functional.cross_entropy(model(x), y).backward()
                        opt.step()
                    torch.cuda.synchronize()
                    t0 = time.time()
                    for _ in range(10):
                        opt.zero_grad()
                        nn.functional.cross_entropy(model(x), y).backward()
                        opt.step()
                    torch.cuda.synchronize()
                    dt = (time.time() - t0) / 10
                    mem = torch.cuda.max_memory_allocated(dev) / 1e9
                    n_train = report["resolutions"][str(size)]["splits"]["train"]["N"]
                    ep_time = dt * np.ceil(n_train / bs)
                    key = f"{size}_bs{bs}"
                    report["gpu"][key] = {"vram_gb": mem, "s_per_iter": dt,
                                          "est_s_per_epoch": float(ep_time)}
                    print(f"{size}^3 bs={bs:3d} | VRAM {mem:5.2f} Go | "
                          f"{dt * 1000:6.1f} ms/iter | ~{ep_time:5.1f} s/epoch "
                          "(petit CNN, estimation)")
                except RuntimeError as e:
                    print(f"{size}^3 bs={bs} : echec ({str(e)[:60]})")
                    torch.cuda.empty_cache()
                del model, opt, x, y
except ImportError:
    print("torch non installe : test GPU saute.")
    report["gpu"] = "torch_missing"

with open(f"{OUT}/report.json", "w") as f:
    json.dump(report, f, indent=2)
print(f"\nRapport : {OUT}/report.json | images : {OUT}/*.png")
print("Envoie-moi tout le texte ci-dessus.")
