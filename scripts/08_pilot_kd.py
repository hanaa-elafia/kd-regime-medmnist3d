"""Pilote KD : 1 teacher x 1 student x 7 methodes x seeds, sur un dataset MedMNIST 3D.

Meme protocole que 06_screen_models.py : 80 epochs, AdamW lr 1e-3 (wd 1e-4),
batch 32, cosine, normalisation avec les stats du TRAIN, checkpoint choisi sur la
NLL de validation, augmentation sans flips (active par defaut ici).
Le TEST n'est JAMAIS charge.

Methodes (guide, section 7) :
  student                      CE seule (alpha = 0), reference sans KD
  vanilla                      T = 4, alpha = 0.5, fixes
  static_T{2,4,8}_a{0.2,0.5,0.8}   grille 3x3 ; le meilleur point (NLL val moyenne) = Static KD
  lin_T / lin_A / lin_TA       schedule lineaire : T 8->2, alpha 0.8->0.2, ou les deux
  fts                          T fuzzy, alpha = 0.5
  fas                          T = 4, alpha fuzzy
  dfs                          T fuzzy + alpha fuzzy (= FTS + FAS)

Etapes :
  1) --stage teacher   entraine et sauvegarde le teacher (seed 0)
  2) --stage run       lance des methodes (un job par GPU, voir les exemples)
  3) --stage summary   tableaux + verifications (pas de GPU necessaire)

Exemples (dossier du projet, env kd-medmnist) :
  # test rapide du pipeline (2 epochs, 1 seed) - a faire en premier
  python scripts/08_pilot_kd.py --stage teacher --smoke
  python scripts/08_pilot_kd.py --stage run --smoke --methods all
  python scripts/08_pilot_kd.py --stage summary --smoke

  # pilote complet
  CUDA_VISIBLE_DEVICES=0 python scripts/08_pilot_kd.py --stage teacher
  nohup env CUDA_VISIBLE_DEVICES=0 python scripts/08_pilot_kd.py --stage run \
      --methods student,vanilla,lin_T,lin_A,lin_TA,fts,fas,dfs --tag _gpu0 \
      > outputs/log_pilot_gpu0.txt 2>&1 &
  nohup env CUDA_VISIBLE_DEVICES=1 python scripts/08_pilot_kd.py --stage run \
      --methods static --tag _gpu1 > outputs/log_pilot_gpu1.txt 2>&1 &
  python scripts/08_pilot_kd.py --stage summary

Sorties dans outputs/pilot_<dataset>/ :
  teacher_<teacher>.pt, teacher_results.csv, results<tag>.csv (1 ligne par run),
  epochs<tag>.csv (1 ligne par epoch : loss, T, alpha, ecart teacher-student...)
"""
import argparse
import copy
import glob
import os
import time

import numpy as np
import pandas as pd

# ===================================================================== BLOC FUZZY
# Bornes communes a Static, Linear et fuzzy (a figer dans le protocole).
T_MIN, T_MAX = 2.0, 8.0
A_MIN, A_MAX = 0.2, 0.8
T0, A0 = 4.0, 0.5            # valeurs de Vanilla ; aussi la valeur "fixe" de FTS / FAS
GRID_T = (2.0, 4.0, 8.0)
GRID_A = (0.2, 0.5, 0.8)
STATIC = [f"static_T{int(t)}_a{a}" for t in GRID_T for a in GRID_A]
NON_STATIC = ["student", "vanilla", "lin_T", "lin_A", "lin_TA", "fts", "fas", "dfs"]
# Test des bornes (hors protocole) : 3 points au-dela de la grille de base.
STATIC_EXT = ["static_T8_a0.9", "static_T16_a0.8", "static_T16_a0.9"]

# Ensembles fuzzy : partition triangulaire Low / Medium / High sur [0, 1].
U = np.linspace(0.0, 1.0, 101)
DEFUZZ = "centroid"   # "centroid" (propose) ou "wavg" (moyenne ponderee de singletons 0 / 0.5 / 1)


def m_low(x):
    return np.clip((0.5 - x) / 0.5, 0.0, 1.0)


def m_med(x):
    return np.clip(1.0 - np.abs(x - 0.5) / 0.5, 0.0, 1.0)


def m_high(x):
    return np.clip((x - 0.5) / 0.5, 0.0, 1.0)


MEMB = {"low": m_low, "med": m_med, "high": m_high}
OUT_SETS = {k: f(U) for k, f in MEMB.items()}

# 9 regles : (progression, ecart) -> sortie. Meme base pour T et pour alpha.
RULES = {
    ("low", "low"): "high", ("low", "med"): "high", ("low", "high"): "high",
    ("med", "low"): "med", ("med", "med"): "med", ("med", "high"): "high",
    ("high", "low"): "low", ("high", "med"): "med", ("high", "high"): "med",
}


def fuzzy_u(p, g):
    """Controleur fuzzy (Mamdani, min / max, centroide).
    p : progression t/E dans [0, 1] ; g : ecart teacher-student relatif dans [0, 1].
    Retourne u dans [0, 1] (0 = sortie Low, 1 = sortie High)."""
    mp = {k: float(f(p)) for k, f in MEMB.items()}
    mg = {k: float(f(g)) for k, f in MEMB.items()}
    agg = np.zeros_like(U)
    for (lp, lg), out in RULES.items():
        w = min(mp[lp], mg[lg])
        agg = np.maximum(agg, np.minimum(w, OUT_SETS[out]))
    if DEFUZZ == "wavg":
        # force d'activation maximale de chaque ensemble de sortie, centres 0 / 0.5 / 1
        w = {k: 0.0 for k in OUT_SETS}
        for (lp, lg), out in RULES.items():
            w[out] = max(w[out], min(mp[lp], mg[lg]))
        tot = sum(w.values())
        return (0.5 * w["med"] + 1.0 * w["high"]) / tot if tot > 0 else 0.5
    s = agg.sum()
    return float((U * agg).sum() / s) if s > 0 else 0.5


def get_TA(method, ep, E, g):
    """(T, alpha) pour l'epoch ep (0-indexee) sur E epochs.
    g = ecart teacher-student relatif (utilise seulement par le fuzzy).
    La progression p = ep / (E - 1) va de 0 a 1, pour que le schedule atteigne ses bornes."""
    p = ep / (E - 1) if E > 1 else 0.0
    if method == "student":
        return None, 0.0
    if method == "vanilla":
        return T0, A0
    if method.startswith("static_"):
        _, t, a = method.split("_")
        return float(t[1:]), float(a[1:])

    def lin(a, b):
        return a + p * (b - a)

    if method == "lin_T":
        return lin(T_MAX, T_MIN), A0
    if method == "lin_A":
        return T0, lin(A_MAX, A_MIN)
    if method == "lin_TA":
        return lin(T_MAX, T_MIN), lin(A_MAX, A_MIN)
    u = fuzzy_u(p, g)
    t_f = T_MIN + u * (T_MAX - T_MIN)
    a_f = A_MIN + u * (A_MAX - A_MIN)
    if method == "fts":
        return t_f, A0
    if method == "fas":
        return T0, a_f
    if method == "dfs":
        return t_f, a_f
    raise ValueError(f"methode inconnue : {method}")
# ================================================================= FIN BLOC FUZZY


def expand_methods(spec):
    if spec == "all":
        return NON_STATIC + STATIC
    out = []
    for m in spec.split(","):
        if m == "static":
            out += STATIC
        elif m == "static_ext":
            out += STATIC_EXT
        else:
            out.append(m)
    for m in out:
        assert m in NON_STATIC + STATIC + STATIC_EXT, f"methode inconnue : {m}"
    return out


# =================================================================== PARTIE TORCH
def main_torch(args):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    BATCH, LR = 32, 1e-3
    E = args.epochs

    # ------------------------------------------------ modeles (copie de 06)
    def build_models(n_cls):
        def plain_cnn(ch):
            layers, cin = [], 1
            for c in ch:
                layers += [nn.Conv3d(cin, c, 3, padding=1), nn.BatchNorm3d(c),
                           nn.ReLU(inplace=True), nn.MaxPool3d(2)]
                cin = c
            return nn.Sequential(*layers, nn.AdaptiveAvgPool3d(1), nn.Flatten(),
                                 nn.Linear(cin, n_cls))

        class BasicBlock(nn.Module):
            def __init__(self, cin, cout, stride):
                super().__init__()
                self.c1 = nn.Conv3d(cin, cout, 3, stride, 1, bias=False)
                self.b1 = nn.BatchNorm3d(cout)
                self.c2 = nn.Conv3d(cout, cout, 3, 1, 1, bias=False)
                self.b2 = nn.BatchNorm3d(cout)
                self.sc = None
                if stride != 1 or cin != cout:
                    self.sc = nn.Sequential(
                        nn.Conv3d(cin, cout, 1, stride, bias=False),
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
                                 nn.Linear(cin, n_cls))

        return {
            "S1": lambda: plain_cnn((2, 4, 8, 16)),
            "S2": lambda: plain_cnn((4, 8, 16, 32)),
            "S3": lambda: plain_cnn((8, 16, 32, 64)),
            "res_w16_b1": lambda: resnet3d((16, 32, 64, 128), (1, 1, 1, 1)),
            "plain_w32": lambda: plain_cnn((32, 64, 128, 256)),
            "res_w32_b1": lambda: resnet3d((32, 64, 128, 256), (1, 1, 1, 1)),
            "res_w32_b2": lambda: resnet3d((32, 64, 128, 256), (2, 2, 2, 2)),
        }

    # ------------------------------------------------ donnees (train et val SEULEMENT)
    def load_data(flag, size):
        import medmnist
        from medmnist import INFO
        info = INFO[flag]
        DataClass = getattr(medmnist, info["python_class"])

        def load(split):
            ds = DataClass(split=split, download=True, size=size)
            x = torch.from_numpy(ds.imgs).float().div(255.0)
            x = x.unsqueeze(1) if x.ndim == 4 else x
            return x, torch.from_numpy(ds.labels.flatten()).long()

        return load("train"), load("val"), len(info["label"])

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
        """Translation +-2 voxels, echelle/decalage d'intensite, bruit. PAS de flips."""
        B, _, D, H, W = x.shape
        xp = F.pad(x, (2,) * 6, mode="replicate")
        out = torch.empty_like(x)
        sh = torch.randint(0, 5, (B, 3)).tolist()
        for i, (a, b, c) in enumerate(sh):
            out[i] = xp[i, :, a:a + D, b:b + H, c:c + W]
        scale = 1 + 0.1 * (torch.rand(B, 1, 1, 1, 1, device=x.device) * 2 - 1)
        shift = 0.1 * torch.randn(B, 1, 1, 1, 1, device=x.device)
        return out * scale + shift + 0.05 * torch.randn_like(out)

    def auc_score(probs, y, n_cls):
        try:
            if n_cls == 2:
                return roc_auc_score(y, probs[:, 1])
            return roc_auc_score(y, probs, multi_class="ovr", average="macro",
                                 labels=list(range(n_cls)))
        except ValueError:
            return float("nan")

    # ------------------------------------------------ un entrainement
    def train_run(make, method, seed, teacher, D, keep_state=False):
        """Meme boucle pour toutes les methodes. Meme seed => meme init, meme ordre des
        donnees, meme augmentation (le teacher n'utilise aucun tirage aleatoire)."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = make().to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=args.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, E)
        best, state, ep_rows, G_hist = None, None, [], []
        t_run = time.time()
        for ep in range(E):
            t_ep = time.time()
            # ecart teacher-student relatif (causal : epoch precedente / epoch 1)
            g = 1.0 if ep == 0 else float(
                np.clip(G_hist[-1] / max(G_hist[0], 1e-8), 0.0, 1.0))
            T, a = get_TA(method, ep, E, g)
            lr_now = sched.get_last_lr()[0]
            model.train()
            perm = torch.randperm(len(D["xtr"]), device=dev)
            loss_sum = torch.zeros((), device=dev)
            gap_sum = torch.zeros((), device=dev)
            n_seen = 0
            for i in range(0, len(perm), BATCH):
                idx = perm[i:i + BATCH]
                if len(idx) < 2:
                    continue
                opt.zero_grad()
                xb = augment(D["xtr"][idx]) if not args.no_aug else D["xtr"][idx]
                out = model(xb)
                ce = F.cross_entropy(out, D["ytr"][idx])
                if teacher is None:
                    loss = ce
                else:
                    with torch.no_grad():
                        tl = teacher(xb)
                        # ecart mesure a T = 1, pour ne pas dependre du T du schedule
                        gap_sum += F.kl_div(F.log_softmax(out.detach(), 1),
                                            F.softmax(tl, 1),
                                            reduction="batchmean") * len(idx)
                    kd = F.kl_div(F.log_softmax(out / T, 1), F.softmax(tl / T, 1),
                                  reduction="batchmean") * T * T
                    loss = (1 - a) * ce + a * kd
                loss.backward()
                opt.step()
                loss_sum += loss.detach() * len(idx)
                n_seen += len(idx)
            sched.step()
            G = gap_sum.item() / n_seen if teacher is not None else float("nan")
            G_hist.append(G)

            logits = val_logits(model, D["xva"])
            nll = F.cross_entropy(logits, D["yva"]).item()
            probs = logits.softmax(1)
            pred = probs.argmax(1).numpy()
            acc = float((pred == D["yva_np"]).mean())
            f1 = f1_score(D["yva_np"], pred, average="macro", zero_division=0)
            bacc = balanced_accuracy_score(D["yva_np"], pred)
            ep_rows.append({
                "method": method, "seed": seed, "epoch": ep + 1,
                "train_loss": loss_sum.item() / n_seen, "val_loss": nll,
                "val_accuracy": acc, "val_macro_f1": f1,
                "val_balanced_accuracy": bacc,
                "T": T if T is not None else float("nan"), "alpha": a,
                "gap_kl_t1": G, "gap_rel_used": g, "learning_rate": lr_now,
                "epoch_time_s": time.time() - t_ep})
            if best is None or nll < best["nll"]:
                best = {"nll": nll, "ep": ep + 1, "f1": f1, "bacc": bacc, "acc": acc,
                        "ece": ece_score(probs, D["yva"]),
                        "auc": auc_score(probs.numpy(), D["yva_np"], D["n_cls"])}
                if keep_state:
                    state = copy.deepcopy(model.state_dict())
        dt = time.time() - t_run
        ep_df = pd.DataFrame(ep_rows)
        diverged = bool(not np.isfinite(ep_df["train_loss"]).all())
        Tv, av = ep_df["T"], ep_df["alpha"]
        summary = {"val_nll": best["nll"], "val_macro_f1": best["f1"],
                   "val_bal_acc": best["bacc"], "val_acc": best["acc"],
                   "val_ece": best["ece"], "val_auc": best["auc"],
                   "best_epoch": best["ep"], "train_time_s": dt,
                   "T_min": Tv.min(), "T_max": Tv.max(),
                   "alpha_min": av.min(), "alpha_max": av.max(),
                   "diverged": diverged}
        del model, opt
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        return summary, ep_df, state

    # ------------------------------------------------ preparation
    flag = args.dataset.lower()
    out_dir = f"outputs/pilot_{flag}" + ("_smoke" if args.smoke else "")
    os.makedirs(out_dir, exist_ok=True)
    (xtr, ytr), (xva, yva), n_cls = load_data(flag, args.size)
    mean, std = xtr.mean().item(), xtr.std().item()   # stats du TRAIN seulement
    xtr, xva = (xtr - mean) / std, (xva - mean) / std
    D = {"xtr": xtr.to(dev), "ytr": ytr.to(dev), "xva": xva, "yva": yva,
         "yva_np": yva.numpy(), "n_cls": n_cls}
    models = build_models(n_cls)
    assert args.teacher in models and args.student in models
    print(f"Dataset {flag} | {n_cls} classes | {args.size}^3 | {E} epochs | "
          f"aug={not args.no_aug} wd={args.wd} | device={dev}")
    print(f"train N={len(ytr)} val N={len(yva)} | mean={mean:.4f} std={std:.4f}")
    print(f"teacher={args.teacher} student={args.student}\n", flush=True)
    t_path = f"{out_dir}/teacher_{args.teacher}.pt"
    ident = {"dataset": flag, "teacher": args.teacher, "student": args.student}

    # ------------------------------------------------ etape 1 : teacher
    if args.stage == "teacher":
        if os.path.exists(t_path) and not args.overwrite:
            raise SystemExit(f"{t_path} existe deja (ajoute --overwrite pour refaire).")
        summ, ep_df, state = train_run(models[args.teacher], "student", 0, None, D,
                                       keep_state=True)
        torch.save(state, t_path)
        pd.DataFrame([{**ident, "seed": 0, **summ}]).to_csv(
            f"{out_dir}/teacher_results.csv", index=False)
        print(f"Teacher {args.teacher} | NLL={summ['val_nll']:.4f} "
              f"F1={summ['val_macro_f1']:.4f} AUC={summ['val_auc']:.4f} "
              f"best_ep={summ['best_epoch']} | {summ['train_time_s']:.0f}s")
        print(f"Checkpoint : {t_path}")
        print("Envoie-moi tout le texte ci-dessus.")
        return

    # ------------------------------------------------ etape 2 : methodes
    methods = expand_methods(args.methods)
    res_path = f"{out_dir}/results{args.tag}.csv"
    ep_path = f"{out_dir}/epochs{args.tag}.csv"
    if (os.path.exists(res_path) or os.path.exists(ep_path)) and not args.overwrite:
        raise SystemExit(f"{res_path} existe deja : change --tag ou ajoute --overwrite.")
    for pth in (res_path, ep_path):
        if os.path.exists(pth):
            os.remove(pth)
    assert os.path.exists(t_path), f"teacher introuvable : lance --stage teacher d'abord"
    teacher = models[args.teacher]().to(dev)
    teacher.load_state_dict(torch.load(t_path, map_location=dev))
    teacher.eval()
    for prm in teacher.parameters():
        prm.requires_grad_(False)
    tl = val_logits(teacher, D["xva"])
    print(f"Teacher recharge : val NLL={F.cross_entropy(tl, D['yva']).item():.4f}\n",
          flush=True)

    rows, first_ep = [], True
    for method in methods:
        for seed in range(args.seeds):
            summ, ep_df, _ = train_run(
                models[args.student], method, seed,
                None if method == "student" else teacher, D)
            rows.append({**ident, "method": method, "seed": seed, **summ})
            pd.DataFrame(rows).to_csv(res_path, index=False)
            ep_df.assign(**ident).to_csv(ep_path, mode="a", header=first_ep, index=False)
            first_ep = False
            tr = (f"T=[{summ['T_min']:.2f},{summ['T_max']:.2f}] "
                  f"a=[{summ['alpha_min']:.2f},{summ['alpha_max']:.2f}]")
            print(f"{method:<18} seed={seed} | NLL={summ['val_nll']:.4f} "
                  f"F1={summ['val_macro_f1']:.4f} AUC={summ['val_auc']:.4f} "
                  f"ECE={summ['val_ece']:.4f} best_ep={summ['best_epoch']} | {tr} | "
                  f"{summ['train_time_s']:.0f}s"
                  + (" | DIVERGE" if summ["diverged"] else ""), flush=True)
    print(f"\nFichiers : {res_path} , {ep_path}")
    print("Envoie-moi tout le texte ci-dessus.")


# ===================================================================== RESUME
def summarize(args):
    flag = args.dataset.lower()
    out_dir = f"outputs/pilot_{flag}" + ("_smoke" if args.smoke else "")
    rf = sorted(glob.glob(f"{out_dir}/results*.csv"))
    ef = sorted(glob.glob(f"{out_dir}/epochs*.csv"))
    assert rf and ef, f"aucun resultat dans {out_dir}"
    res = pd.concat([pd.read_csv(f) for f in rf], ignore_index=True)
    ep = pd.concat([pd.read_csv(f) for f in ef], ignore_index=True)
    pd.set_option("display.width", 200)

    tpath = f"{out_dir}/teacher_results.csv"
    if os.path.exists(tpath):
        t = pd.read_csv(tpath).iloc[0]
        print(f"Teacher {t['teacher']} : NLL={t['val_nll']:.4f} F1={t['val_macro_f1']:.4f} "
              f"AUC={t['val_auc']:.4f} ECE={t['val_ece']:.4f} best_ep={int(t['best_epoch'])}")
    print(f"Student {res['student'].iloc[0]} | {res['seed'].nunique()} seeds | "
          f"methodes : {res['method'].nunique()}\n")

    # --- Static : choix sur la validation (NLL moyenne), sans regarder le test
    st = res[res["method"].isin(STATIC)]
    ext = res[res["method"].isin(STATIC_EXT)]
    best_static = None
    if len(st):
        grid = st.groupby("method")[["val_nll", "val_macro_f1"]].mean().sort_values("val_nll")
        print("=== Static KD : grille (moyenne sur les seeds, validation) ===")
        print(grid.round(4).to_string())
        best_static = grid.index[0]
        _, tb, ab = best_static.split("_")
        tb, ab = float(tb[1:]), float(ab[1:])
        print(f"\nMeilleur point : T={tb:g}, alpha={ab:g}")
        for nom, v, lo, hi in (("T", tb, T_MIN, T_MAX), ("alpha", ab, A_MIN, A_MAX)):
            if v in (lo, hi):
                print(f"  ATTENTION : {nom} est sur une borne ({v:g}). Avec 3 niveaux "
                      "par axe, le test est faible ; si c'est net, envisager d'elargir "
                      "les bornes AVANT le gel du protocole.")
        print()
        if len(ext):
            ref = grid["val_nll"].iloc[0]
            sd_ref = st[st["method"] == best_static]["val_nll"].std()
            print("=== Test des bornes : points au-dela de la grille de base ===")
            print(f"reference = meilleur point de la grille ({best_static}) : NLL={ref:.4f} "
                  f"(ecart-type entre seeds {sd_ref:.4f})")
            e = ext.groupby("method")["val_nll"].agg(["mean", "std"])
            e["diff_vs_ref"] = e["mean"] - ref   # negatif = mieux que la grille de base
            print(e.sort_values("mean").round(4).to_string())
            print("Lecture : un point n'est meilleur que si la difference negative depasse "
                  "nettement l'ecart-type entre seeds.\n")

    # --- tableau par methode (static = meilleur point)
    sel = res[~res["method"].str.startswith("static_") | (res["method"] == best_static)].copy()
    if best_static:
        sel.loc[sel["method"] == best_static, "method"] = "static(best)"
    order = ["student", "vanilla", "static(best)", "lin_T", "lin_A", "lin_TA",
             "fts", "fas", "dfs"]
    order = [m for m in order if m in set(sel["method"])]
    cols = ["val_nll", "val_macro_f1", "val_bal_acc", "val_ece", "val_auc",
            "best_epoch", "train_time_s"]
    agg = sel.groupby("method")[cols].agg(["mean", "std"]).reindex(order).round(4)
    print("=== Resume par methode (validation, 3 seeds : NE PAS en tirer de conclusion) ===")
    print(agg.to_string(), "\n")

    # --- gains apparies par seed contre Student
    base = sel[sel["method"] == "student"].set_index("seed")
    if len(base):
        print("=== Gain contre Student, apparie par seed (moyenne +- ecart-type) ===")
        print("gain_NLL = NLL(student) - NLL(methode) : positif = mieux ; "
              "gain_F1 = F1(methode) - F1(student)")
        rows = []
        for m in order:
            if m == "student":
                continue
            d = sel[sel["method"] == m].set_index("seed")
            c = base.index.intersection(d.index)
            gn = (base.loc[c, "val_nll"] - d.loc[c, "val_nll"])
            gf = (d.loc[c, "val_macro_f1"] - base.loc[c, "val_macro_f1"])
            rows.append({"methode": m, "n_seeds": len(c),
                         "gain_NLL": f"{gn.mean():+.4f} +- {gn.std():.4f}",
                         "gain_F1": f"{gf.mean():+.4f} +- {gf.std():.4f}"})
        print(pd.DataFrame(rows).to_string(index=False), "\n")

    # --- verification des contoleurs : T et alpha dans les bornes, non constants
    print("=== Controle des schedules : plage reelle de T et alpha ===")
    chk = []
    for m in ["lin_T", "lin_A", "lin_TA", "fts", "fas", "dfs"]:
        e = ep[ep["method"] == m]
        if len(e):
            chk.append({"methode": m, "T_min": e["T"].min(), "T_max": e["T"].max(),
                        "alpha_min": e["alpha"].min(), "alpha_max": e["alpha"].max()})
    print(pd.DataFrame(chk).round(3).to_string(index=False))
    print(f"(bornes du protocole : T [{T_MIN:g},{T_MAX:g}], alpha [{A_MIN:g},{A_MAX:g}])\n")

    print("=== Trajectoires moyennes (sur les seeds) aux epochs cles ===")
    n_ep = int(ep["epoch"].max())
    keys = sorted({1, max(1, n_ep // 8), n_ep // 4, n_ep // 2, 3 * n_ep // 4, n_ep} - {0})
    for m in ["fts", "fas", "dfs"]:
        e = ep[(ep["method"] == m) & (ep["epoch"].isin(keys))]
        if len(e):
            print(f"\n{m} :")
            print(e.groupby("epoch")[["T", "alpha", "gap_kl_t1", "gap_rel_used"]]
                  .mean().round(3).T.to_string())
    print()

    # --- alertes
    bad = res[~np.isfinite(res["val_nll"]) | res["diverged"]]
    if len(bad):
        print("ALERTE : runs divergents ou NaN :")
        print(bad[["method", "seed"]].to_string(index=False))
    else:
        print("Aucun run divergent.")
    print("Envoie-moi tout le texte ci-dessus.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=["teacher", "run", "summary"])
    parser.add_argument("--dataset", default="organmnist3d")
    parser.add_argument("--teacher", default="res_w32_b1")
    parser.add_argument("--student", default="S1")
    parser.add_argument("--methods", default="all",
                        help="liste separee par virgules ; 'static' = les 9 points de la "
                             "grille ; 'all' = tout")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--size", type=int, default=28)
    parser.add_argument("--tag", default="")
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--no_aug", action="store_true",
                        help="desactiver l'augmentation (active par defaut dans le pilote)")
    parser.add_argument("--smoke", action="store_true",
                        help="test rapide : 2 epochs, 1 seed, dossier separe")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--defuzz", default="centroid", choices=["centroid", "wavg"],
                        help="defuzzification du controleur fuzzy (a fixer AVANT de "
                             "regarder les resultats du pilote)")
    a = parser.parse_args()
    DEFUZZ = a.defuzz
    if a.smoke:
        a.epochs, a.seeds = 2, 1
    if a.stage == "summary":
        summarize(a)
    else:
        main_torch(a)
