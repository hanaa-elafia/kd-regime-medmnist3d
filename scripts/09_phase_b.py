"""Phase B : experiences finales, selon la fiche de gel (fiche_gel_protocole_v1.md).

Un dataset par lancement. Memes modeles, memes hyperparametres et meme boucle
d'entrainement que 08_pilot_kd.py (80 epochs, AdamW lr 1e-3 wd 1e-4, batch 32,
cosine, augmentation sans flips, checkpoint = meilleure NLL de validation).

Ce que ce script ajoute au pilote :
  - le TEST est evalue UNE SEULE FOIS, uniquement pour les runs retenus : JAMAIS pour les
    points de la grille Static, JAMAIS pour les teachers pendant leur entrainement (leurs
    metriques de test se calculent a l'analyse finale, depuis leur checkpoint). Le point
    Static retenu est evalue au test apres sa selection sur la validation, depuis les
    checkpoints sauvegardes de la grille. Les metriques de test sont ecrites dans les CSV
    mais JAMAIS affichees dans les logs ;
  - gardes de reprise : arret si des jobs sont dupliques ou si le nombre d'epochs differe ;
  - le point Static est choisi sur la validation seulement (seeds 0 a 2) ;
  - reprise automatique : un job deja present dans un results*.csv est saute.

Etapes :
  --stage teachers   entraine et sauvegarde les teachers (seed 0), une seule fois par dataset
  --stage run        lance un plan :
        students  Student seul : seeds 0-4 (S1, S2, S3) + seeds 5-14 pour S1 (pivot)
        primary   7 methodes de KD x seeds 0-4, grille Static (seeds 0-2), puis le meilleur
                  point Static sur les seeds 3-4
        pivot     couple pivot res_w32_b2 x S1 : 7 methodes + Static au meilleur point,
                  seeds 5-14
        b2a       phase B2a (addendum_B2a_v1.md) : Student seul et Vanilla sur les 27
                  conditions, seeds 15-24, MEMES teachers que la Phase B (lus dans
                  outputs/phaseB_<dataset>/) ; resultats dans outputs/phaseB2a_<dataset>/
  --stage status     avancement et verifications (validation seulement, jamais le test)

Sorties dans outputs/phaseB_<dataset>/ :
  teacher_<nom>.pt, teachers_results.csv, results<tag>.csv, epochs<tag>.csv

Controle de coherence avec le pilote (80 epochs, AUCUN calcul de test, dossier _check) :
  python scripts/09_phase_b.py --stage run --plan students --dataset organmnist3d --check --students S1
  -> comparer les seeds 0 a 2 de S1 avec le pilote (NLL de validation : 0.0478, 0.0453, 0.0599).

Test rapide (2 epochs, dossier separe) :
  python scripts/09_phase_b.py --stage teachers --dataset adrenalmnist3d --smoke --teachers res_w16_b1,res_w32_b2
  python scripts/09_phase_b.py --stage run --plan students --dataset adrenalmnist3d --smoke
  python scripts/09_phase_b.py --stage run --plan primary --dataset adrenalmnist3d --smoke --teachers res_w32_b2 --students S1
  python scripts/09_phase_b.py --stage run --plan pivot --dataset adrenalmnist3d --smoke
  python scripts/09_phase_b.py --stage status --dataset adrenalmnist3d --smoke
"""
import argparse
import copy
import glob
import os
import time

import numpy as np
import pandas as pd

# ================================================================== PLAN GELE
T_MIN, T_MAX = 2.0, 8.0
A_MIN, A_MAX = 0.2, 0.8
T0, A0 = 4.0, 0.5
GRID_T = (2.0, 4.0, 8.0)
GRID_A = (0.2, 0.5, 0.8)
STATIC = [f"static_T{int(t)}_a{a}" for t in GRID_T for a in GRID_A]
KD_BASE = ["vanilla", "lin_T", "lin_A", "lin_TA", "fts", "fas", "dfs"]
ALL_TEACHERS = ["res_w16_b1", "res_w32_b1", "res_w32_b2"]
ALL_STUDENTS = ["S1", "S2", "S3"]
PIVOT_TEACHER, PIVOT_STUDENT = "res_w32_b2", "S1"
EPOCHS, BATCH, LR, WD = 80, 32, 1e-3, 1e-4
DATASETS = ["organmnist3d", "adrenalmnist3d", "fracturemnist3d"]

# ----------------------------------------------------------- controleur fuzzy
LEVELS = ("low", "med", "high")


def m_low(x):
    return float(np.clip((0.5 - x) / 0.5, 0.0, 1.0))


def m_med(x):
    return float(np.clip(1.0 - abs(x - 0.5) / 0.5, 0.0, 1.0))


def m_high(x):
    return float(np.clip((x - 0.5) / 0.5, 0.0, 1.0))


MEMB = {"low": m_low, "med": m_med, "high": m_high}
RULES = {
    ("low", "low"): "high", ("low", "med"): "high", ("low", "high"): "high",
    ("med", "low"): "med", ("med", "med"): "med", ("med", "high"): "high",
    ("high", "low"): "low", ("high", "med"): "med", ("high", "high"): "med",
}


def fuzzy_u(p, g):
    """Moyenne ponderee de singletons (0 / 0.5 / 1). p = progression, g = ecart relatif."""
    mp = {k: MEMB[k](p) for k in LEVELS}
    mg = {k: MEMB[k](g) for k in LEVELS}
    w = {k: 0.0 for k in LEVELS}
    for (lp, lg), out in RULES.items():
        w[out] = max(w[out], min(mp[lp], mg[lg]))
    tot = sum(w.values())
    return (0.5 * w["med"] + 1.0 * w["high"]) / tot if tot > 0 else 0.5


def get_TA(cfg, ep, E, g):
    """(T, alpha) pour l'epoch ep (0-indexee) sur E epochs ; g = ecart relatif (fuzzy)."""
    p = ep / (E - 1) if E > 1 else 0.0
    if cfg == "student":
        return None, 0.0
    if cfg == "vanilla":
        return T0, A0
    if cfg.startswith("static_"):
        _, t, a = cfg.split("_")
        return float(t[1:]), float(a[1:])

    def lin(a, b):
        return a + p * (b - a)

    if cfg == "lin_T":
        return lin(T_MAX, T_MIN), A0
    if cfg == "lin_A":
        return T0, lin(A_MAX, A_MIN)
    if cfg == "lin_TA":
        return lin(T_MAX, T_MIN), lin(A_MAX, A_MIN)
    u = fuzzy_u(p, g)
    t_f = T_MIN + u * (T_MAX - T_MIN)
    a_f = A_MIN + u * (A_MAX - A_MIN)
    if cfg == "fts":
        return t_f, A0
    if cfg == "fas":
        return T0, a_f
    if cfg == "dfs":
        return t_f, a_f
    raise ValueError(f"configuration inconnue : {cfg}")


# --------------------------------------------------------------------- jobs
def make_jobs(plan, teachers, students):
    """Liste de jobs (teacher, student, label, config, seed). Le plan est celui de la fiche."""
    jobs = []
    if plan == "students":
        for st in students:
            seeds = list(range(5)) + (list(range(5, 15)) if st == PIVOT_STUDENT else [])
            jobs += [("none", st, "student", "student", sd) for sd in seeds]
    elif plan == "primary":
        for t in teachers:
            for st in students:
                jobs += [(t, st, "static_grid", c, sd) for c in STATIC for sd in range(3)]
                jobs += [(t, st, m, m, sd) for m in KD_BASE for sd in range(5)]
                jobs += [(t, st, "static_best", "@best", sd) for sd in (3, 4)]
    elif plan == "pivot":
        if PIVOT_TEACHER in teachers and PIVOT_STUDENT in students:
            t, st = PIVOT_TEACHER, PIVOT_STUDENT
            jobs += [(t, st, m, m, sd) for m in KD_BASE for sd in range(5, 15)]
            jobs += [(t, st, "static_best", "@best", sd) for sd in range(5, 15)]
    elif plan == "b2a":
        seeds = range(15, 25)
        for st in students:
            jobs += [("none", st, "student", "student", sd) for sd in seeds]
        for t in teachers:
            for st in students:
                jobs += [(t, st, "vanilla", "vanilla", sd) for sd in seeds]
    else:
        raise ValueError(plan)
    return jobs


def job_key(t, st, label, cfg, seed):
    return (t, st, label, "" if label == "static_best" else cfg, int(seed))


def out_dir_for(dataset, smoke, check=False, prefix="phaseB"):
    return f"outputs/{prefix}_{dataset}" + ("_smoke" if smoke else "") + ("_check" if check else "")


def read_results(out_dir):
    files = sorted(glob.glob(f"{out_dir}/results*.csv"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def done_keys(res):
    if res.empty:
        return set()
    return {job_key(r.teacher, r.student, r.method, r.config, r.seed)
            for r in res.itertuples()}


def duplicated_keys(res):
    """Jobs presents plusieurs fois dans les resultats (ne doit jamais arriver)."""
    if res.empty:
        return []
    keys = [job_key(r.teacher, r.student, r.method, r.config, r.seed) for r in res.itertuples()]
    seen, dup = set(), []
    for k in keys:
        if k in seen:
            dup.append(k)
        seen.add(k)
    return dup


def resolve_best(res, t, st):
    """Meilleur point Static pour un couple : plus faible NLL de VALIDATION moyenne sur les
    seeds 0 a 2 (egalite : plus petit T, puis plus petit alpha). Le test n'est jamais lu."""
    g = res[(res["teacher"] == t) & (res["student"] == st)
            & (res["method"] == "static_grid") & (res["seed"] < 3)]
    n = g.groupby("config")["seed"].nunique().reindex(STATIC).fillna(0)
    if (n < 3).any():
        manq = [c for c in STATIC if n[c] < 3]
        raise SystemExit(f"Grille Static incomplete pour {t} x {st} (manque : {manq}). "
                         "Lance d'abord le plan 'primary' jusqu'a la fin de la grille.")
    m = g.groupby("config")["val_nll"].mean()

    def key(c):
        _, tt, aa = c.split("_")
        return (round(float(m[c]), 12), float(tt[1:]), float(aa[1:]))

    return min(STATIC, key=key)


def fmt(x):
    return f"{x:.4f}"


# ==================================================================== TORCH
def main_torch(args):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
    from torch.utils.flop_counter import FlopCounterMode

    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    E = args.epochs
    flag = args.dataset
    t_dir = out_dir_for(flag, args.smoke, args.check)          # dossier des teachers (Phase B)
    is_b2a = args.stage == "run" and args.plan == "b2a"
    out_dir = out_dir_for(flag, args.smoke, args.check, "phaseB2a") if is_b2a else t_dir
    os.makedirs(f"{out_dir}/ckpt", exist_ok=True)

    # ---------------------------------------------- modeles (identiques a 06 / 08)
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
                        nn.Conv3d(cin, cout, 1, stride, bias=False), nn.BatchNorm3d(cout))

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
            "res_w32_b1": lambda: resnet3d((32, 64, 128, 256), (1, 1, 1, 1)),
            "res_w32_b2": lambda: resnet3d((32, 64, 128, 256), (2, 2, 2, 2)),
        }

    def complexity(make):
        m = make().eval()
        n_params = sum(p.numel() for p in m.parameters())
        with FlopCounterMode(display=False) as fc:
            m(torch.zeros(1, 1, args.size, args.size, args.size))
        return n_params, fc.get_total_flops() / 1e6

    # ---------------------------------------------- donnees
    def load_data():
        import medmnist
        from medmnist import INFO
        info = INFO[flag]
        DataClass = getattr(medmnist, info["python_class"])

        def load(split):
            ds = DataClass(split=split, download=True, size=args.size)
            x = torch.from_numpy(ds.imgs).float().div(255.0)
            x = x.unsqueeze(1) if x.ndim == 4 else x
            return x, torch.from_numpy(ds.labels.flatten()).long()

        return load("train"), load("val"), load("test"), len(info["label"])

    @torch.no_grad()
    def logits_of(model, x):
        model.eval()
        return torch.cat([model(x[i:i + 64].to(dev)).cpu() for i in range(0, len(x), 64)])

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
        B, _, D_, H, W = x.shape
        xp = F.pad(x, (2,) * 6, mode="replicate")
        out = torch.empty_like(x)
        sh = torch.randint(0, 5, (B, 3)).tolist()
        for i, (a, b, c) in enumerate(sh):
            out[i] = xp[i, :, a:a + D_, b:b + H, c:c + W]
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

    def evaluate(model, x, y_t, y_np, n_cls):
        logits = logits_of(model, x)
        probs = logits.softmax(1)
        pred = probs.argmax(1).numpy()
        return {"nll": F.cross_entropy(logits, y_t).item(),
                "f1": f1_score(y_np, pred, average="macro", zero_division=0),
                "bacc": balanced_accuracy_score(y_np, pred),
                "acc": float((pred == y_np).mean()),
                "ece": ece_score(probs, y_t), "auc": auc_score(probs.numpy(), y_np, n_cls)}

    # ---------------------------------------------- un entrainement
    def train_run(make, cfg, seed, teacher, D, eval_test=True):
        """Meme boucle que le pilote. Meme seed => meme init, meme ordre des donnees, meme
        augmentation. Si eval_test : le test est evalue une seule fois, a la fin, sur le
        meilleur checkpoint de validation. Sinon (grille Static, teachers, mode --check),
        le test n'est PAS touche."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = make().to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, E)
        best, best_state, ep_rows, G_hist = None, None, [], []
        t_run = time.time()
        for ep in range(E):
            t_ep = time.time()
            g = 1.0 if ep == 0 else float(
                np.clip(G_hist[-1] / max(G_hist[0], 1e-8), 0.0, 1.0))
            T, a = get_TA(cfg, ep, E, g)
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
                xb = augment(D["xtr"][idx])
                out = model(xb)
                ce = F.cross_entropy(out, D["ytr"][idx])
                if teacher is None:
                    loss = ce
                else:
                    with torch.no_grad():
                        tl = teacher(xb)
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

            logits = logits_of(model, D["xva"])
            nll = F.cross_entropy(logits, D["yva"]).item()
            probs = logits.softmax(1)
            pred = probs.argmax(1).numpy()
            acc = float((pred == D["yva_np"]).mean())
            f1 = f1_score(D["yva_np"], pred, average="macro", zero_division=0)
            bacc = balanced_accuracy_score(D["yva_np"], pred)
            ep_rows.append({
                "epoch": ep + 1, "train_loss": loss_sum.item() / n_seen, "val_loss": nll,
                "val_accuracy": acc, "val_macro_f1": f1, "val_balanced_accuracy": bacc,
                "T": T if T is not None else float("nan"), "alpha": a,
                "gap_kl_t1": G, "gap_rel_used": g, "learning_rate": lr_now,
                "epoch_time_s": time.time() - t_ep})
            if best is None or nll < best["nll"]:
                best = {"nll": nll, "ep": ep + 1, "f1": f1, "bacc": bacc, "acc": acc,
                        "ece": ece_score(probs, D["yva"]),
                        "auc": auc_score(probs.numpy(), D["yva_np"], D["n_cls"])}
                best_state = copy.deepcopy(model.state_dict())
        dt = time.time() - t_run
        # --- test : une seule evaluation, sur le meilleur checkpoint de validation
        if eval_test:
            model.load_state_dict(best_state)
            te = evaluate(model, D["xte"], D["yte"], D["yte_np"], D["n_cls"])
        else:
            te = {k: float("nan") for k in ("nll", "f1", "bacc", "acc", "ece", "auc")}
        ep_df = pd.DataFrame(ep_rows)
        Tv, av = ep_df["T"], ep_df["alpha"]
        summ = {"val_nll": best["nll"], "val_macro_f1": best["f1"],
                "val_bal_acc": best["bacc"], "val_acc": best["acc"],
                "val_ece": best["ece"], "val_auc": best["auc"],
                "test_nll": te["nll"], "test_macro_f1": te["f1"],
                "test_bal_acc": te["bacc"], "test_acc": te["acc"],
                "test_ece": te["ece"], "test_auc": te["auc"],
                "best_epoch": best["ep"], "train_time_s": dt,
                "T_min": Tv.min(), "T_max": Tv.max(),
                "alpha_min": av.min(), "alpha_max": av.max(),
                "diverged": bool(not np.isfinite(ep_df["train_loss"]).all()), "epochs": E}
        del model, opt
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        return summ, ep_df, best_state

    # ---------------------------------------------- preparation
    (xtr, ytr), (xva, yva), (xte, yte), n_cls = load_data()
    mean, std = xtr.mean().item(), xtr.std().item()   # stats du TRAIN seulement
    xtr, xva, xte = (xtr - mean) / std, (xva - mean) / std, (xte - mean) / std
    D = {"xtr": xtr.to(dev), "ytr": ytr.to(dev), "xva": xva, "yva": yva,
         "yva_np": yva.numpy(), "xte": xte, "yte": yte, "yte_np": yte.numpy(),
         "n_cls": n_cls}
    models = build_models(n_cls)
    comp = {n: complexity(f) for n, f in models.items()}
    print(f"Phase B | {flag} | {n_cls} classes | {args.size}^3 | {E} epochs | device={dev} | "
          f"stage={args.stage}" + (f" plan={args.plan}" if args.stage == "run" else ""))
    print(f"train N={len(ytr)} val N={len(yva)} test N={len(yte)} | "
          f"mean={mean:.4f} std={std:.4f}", flush=True)
    teachers = args.teachers.split(",") if args.teachers else list(ALL_TEACHERS)
    students = args.students.split(",") if args.students else list(ALL_STUDENTS)
    for n in teachers + students:
        assert n in models, f"modele inconnu : {n}"

    # ---------------------------------------------- etape 1 : teachers
    if args.stage == "teachers":
        rows_path = f"{out_dir}/teachers_results.csv"
        for t in teachers:
            t_path = f"{out_dir}/teacher_{t}.pt"
            if os.path.exists(t_path) and not args.overwrite:
                print(f"{t} : deja entraine ({t_path}), saute.")
                continue
            summ, ep_df, state = train_run(models[t], "student", 0, None, D, eval_test=False)
            tmp = t_path + ".tmp"
            torch.save(state, tmp)
            os.replace(tmp, t_path)   # ecriture atomique
            row = {"dataset": flag, "teacher": t, "seed": 0, **summ,
                   "params": comp[t][0], "mflops": comp[t][1]}
            pd.DataFrame([row]).to_csv(rows_path, mode="a",
                                       header=not os.path.exists(rows_path), index=False)
            print(f"Teacher {t} (test non evalue ici) | val NLL={summ['val_nll']:.4f} val F1={summ['val_macro_f1']:.4f} "
                  f"val AUC={summ['val_auc']:.4f} best_ep={summ['best_epoch']} | "
                  f"{summ['train_time_s']:.0f}s | {t_path}", flush=True)
        print("Teachers termines.")
        return

    # ---------------------------------------------- etape 2 : plan
    res_path = f"{out_dir}/results{args.tag}.csv"
    ep_path = f"{out_dir}/epochs{args.tag}.csv"
    jobs = make_jobs(args.plan, teachers, students)
    if args.no_students:   # evite de lancer les Student seuls sur les deux GPU a la fois
        jobs = [j for j in jobs if j[0] != "none"]
    res0 = read_results(out_dir)
    dups = duplicated_keys(res0)
    if dups:
        raise SystemExit(f"ARRET : {len(dups)} job(s) en double dans les resultats, ex. {dups[:3]}. "
                         "Verifie qu'un seul processus travaille sur chaque job.")
    if not res0.empty and "epochs" in res0.columns and (res0["epochs"] != E).any():
        raise SystemExit(f"ARRET : des resultats existent avec un autre nombre d'epochs "
                         f"({sorted(res0['epochs'].unique())}) que --epochs {E}.")
    done = done_keys(res0)
    todo = [j for j in jobs if job_key(*j) not in done]
    print(f"Jobs du plan : {len(jobs)} | deja faits : {len(jobs) - len(todo)} | "
          f"a faire : {len(todo)}", flush=True)

    def append(path, df):
        df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)

    best_cache, cur_t, cur_teacher = {}, None, None
    for n_done, (t, st, label, cfg, sd) in enumerate(todo, 1):
        if label == "static_best":
            if (t, st) not in best_cache:
                res = read_results(out_dir)
                best_cache[(t, st)] = resolve_best(res, t, st)
                # les seeds 0 a 2 du point retenu sont reprises de la grille (pas relancees)
                for s0 in range(3):
                    k = job_key(t, st, "static_best", "", s0)
                    if k in done:
                        continue
                    src = res[(res["teacher"] == t) & (res["student"] == st)
                              & (res["method"] == "static_grid")
                              & (res["config"] == best_cache[(t, st)])
                              & (res["seed"] == s0)]
                    assert len(src) == 1, f"ligne de grille introuvable : {t} {st} seed {s0}"
                    row = src.iloc[0].to_dict()
                    row.update({"method": "static_best", "reused": True})
                    if not args.check:
                        # le point est retenu (choisi sur la validation) : premier et seul
                        # calcul du test pour ce run, depuis le checkpoint de la grille
                        ck = f"{out_dir}/ckpt/{t}_{st}_{best_cache[(t, st)]}_s{s0}.pt"
                        assert os.path.exists(ck), f"checkpoint introuvable : {ck}"
                        mdl = models[st]().to(dev)
                        mdl.load_state_dict(torch.load(ck, map_location=dev))
                        te = evaluate(mdl, D["xte"], D["yte"], D["yte_np"], D["n_cls"])
                        row.update({"test_nll": te["nll"], "test_macro_f1": te["f1"],
                                    "test_bal_acc": te["bacc"], "test_acc": te["acc"],
                                    "test_ece": te["ece"], "test_auc": te["auc"]})
                        del mdl
                    append(res_path, pd.DataFrame([row]))
                    done.add(k)
                print(f"Static best pour {t} x {st} : {best_cache[(t, st)]} "
                      "(choisi sur la NLL de validation, seeds 0 a 2)", flush=True)
            cfg = best_cache[(t, st)]
        if t != cur_t:
            cur_teacher, cur_t = None, t
            if t != "none":
                cur_teacher = models[t]().to(dev)
                t_path = f"{t_dir}/teacher_{t}.pt"
                assert os.path.exists(t_path), f"teacher introuvable : {t_path}"
                cur_teacher.load_state_dict(torch.load(t_path, map_location=dev))
                cur_teacher.eval()
                for prm in cur_teacher.parameters():
                    prm.requires_grad_(False)
        is_grid = label == "static_grid"
        summ, ep_df, state = train_run(models[st], cfg, sd, cur_teacher, D,
                                       eval_test=(not args.check) and (not is_grid))
        if is_grid:   # le test n'est pas touche : on garde le checkpoint pour plus tard
            ck = f"{out_dir}/ckpt/{t}_{st}_{cfg}_s{sd}.pt"
            torch.save(state, ck + ".tmp")
            os.replace(ck + ".tmp", ck)
        row = {"dataset": flag, "teacher": t, "student": st, "method": label,
               "config": cfg, "seed": sd, **summ,
               "student_params": comp[st][0], "student_mflops": comp[st][1],
               "teacher_params": comp[t][0] if t != "none" else 0,
               "teacher_mflops": comp[t][1] if t != "none" else 0, "reused": False}
        append(res_path, pd.DataFrame([row]))
        append(ep_path, ep_df.assign(dataset=flag, teacher=t, student=st,
                                     method=label, config=cfg, seed=sd))
        done.add(job_key(t, st, label, cfg, sd))
        # NB : seules des metriques de VALIDATION sont affichees (le test reste dans le CSV)
        print(f"[{n_done}/{len(todo)}] {t:<11} {st} {label:<12} {cfg:<16} seed={sd} | "
              f"val NLL={summ['val_nll']:.4f} F1={summ['val_macro_f1']:.4f} "
              f"best_ep={summ['best_epoch']} | {summ['train_time_s']:.0f}s"
              + (" | DIVERGE" if summ["diverged"] else ""), flush=True)
    print("\nPlan termine.")


# ==================================================================== STATUS
def status_b2a(args):
    out_dir = out_dir_for(args.dataset, args.smoke, args.check, "phaseB2a")
    res = read_results(out_dir)
    exp = {job_key(*j) for j in make_jobs("b2a", ALL_TEACHERS, ALL_STUDENTS)}
    done = done_keys(res)
    print(f"Dossier : {out_dir}")
    print(f"Plan b2a : {len(exp & done)} / {len(exp)} ({100 * len(exp & done) / len(exp):.0f} %)")
    if res.empty:
        return
    dups = duplicated_keys(res)
    print(f"Jobs en double : {len(dups)}" + (f" (ex. {dups[:3]})" if dups else ""))
    print(f"Epochs presentes : {sorted(int(x) for x in res['epochs'].dropna().unique())}")
    print(f"Jobs hors plan : {len(done - exp)}")
    bad = res[res["diverged"].astype(bool) | ~np.isfinite(res["val_nll"])]
    print(f"Runs divergents ou NaN : {len(bad)}")
    print("Seeds presentes :", sorted(int(x) for x in res["seed"].unique()))
    print("\n(Aucune metrique de test n'est affichee ici.)")


def status(args):
    if getattr(args, "plan", None) == "b2a":
        return status_b2a(args)
    out_dir = out_dir_for(args.dataset, args.smoke, args.check)
    res = read_results(out_dir)
    done = done_keys(res)
    print(f"Dossier : {out_dir}")
    print("Teachers sauvegardes :",
          [t for t in ALL_TEACHERS if os.path.exists(f"{out_dir}/teacher_{t}.pt")] or "aucun")
    tp = f"{out_dir}/teachers_results.csv"
    if os.path.exists(tp):
        tr = pd.read_csv(tp)
        print("Teachers (validation) :")
        print(tr[["teacher", "val_nll", "val_macro_f1", "val_auc", "best_epoch"]]
              .round(4).to_string(index=False))
    print()
    tot_exp = tot_done = 0
    for plan in ("students", "primary", "pivot"):
        jobs = make_jobs(plan, ALL_TEACHERS, ALL_STUDENTS)
        exp = {job_key(*j) for j in jobs}
        if plan == "primary":   # lignes static_best reprises de la grille (seeds 0 a 2)
            exp |= {job_key(t, s, "static_best", "", sd)
                    for t in ALL_TEACHERS for s in ALL_STUDENTS for sd in range(3)}
        d = len(exp & done)
        tot_exp += len(exp)
        tot_done += d
        print(f"Plan {plan:<9}: {d:4d} / {len(exp):4d} ({100 * d / max(len(exp), 1):.0f} %)")
    print(f"Total        : {tot_done:4d} / {tot_exp:4d}\n")
    if res.empty:
        return
    dups = duplicated_keys(res)
    print(f"Jobs en double : {len(dups)}" + (f" (ex. {dups[:3]})" if dups else ""))
    if "epochs" in res.columns:
        print(f"Epochs presentes dans les resultats : {sorted(int(x) for x in res['epochs'].dropna().unique())}")
    prim = res[res["method"].isin(["static_grid"] + KD_BASE) & (res["seed"] < 5)]
    if len(prim):
        print("Primary par couple (seeds 0 a 4, hors lignes reprises ; 62 runs par couple a la fin) :")
        c = (prim.groupby(["teacher", "student"]).size().rename("runs").reset_index())
        print(c.to_string(index=False))
    sb = res[(res["method"] == "static_best") & (~res["reused"].astype(bool))]
    if len(sb):
        print("\nPoint Static retenu par couple (validation) :")
        print(sb.groupby(["teacher", "student"])["config"].first().to_string())
    bad = res[res["diverged"].astype(bool) | ~np.isfinite(res["val_nll"])]
    print(f"\nRuns divergents ou NaN : {len(bad)}")
    if len(bad):
        print(bad[["teacher", "student", "method", "config", "seed"]].to_string(index=False))
    fz = res[res["method"].isin(["fts", "fas", "dfs", "lin_T", "lin_A", "lin_TA"])]
    if len(fz):
        print("\nPlages reelles de T et alpha (bornes : T [2, 8], alpha [0.2, 0.8]) :")
        print(fz.groupby("method")[["T_min", "T_max", "alpha_min", "alpha_max"]]
              .agg({"T_min": "min", "T_max": "max", "alpha_min": "min", "alpha_max": "max"})
              .round(3).to_string())
    print("\n(Aucune metrique de test n'est affichee ici : l'analyse du test se fait une fois, "
          "a la fin, avec le script d'analyse.)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=["teachers", "run", "status"])
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--plan", choices=["students", "primary", "pivot", "b2a"],
                        help="obligatoire avec --stage run")
    parser.add_argument("--teachers", default="", help="liste, defaut : les 3")
    parser.add_argument("--students", default="", help="liste, defaut : les 3")
    parser.add_argument("--no_students", action="store_true",
                        help="plan b2a : ne pas lancer les Student seuls (pour le 2e GPU)")
    parser.add_argument("--tag", default="")
    parser.add_argument("--size", type=int, default=28)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--smoke", action="store_true",
                        help="test rapide : 2 epochs, dossier separe")
    parser.add_argument("--check", action="store_true",
                        help="controle de coherence : 80 epochs, dossier _check, AUCUN test")
    parser.add_argument("--overwrite", action="store_true",
                        help="re-entrainer un teacher deja sauvegarde (a eviter)")
    a = parser.parse_args()
    if a.smoke and a.check:
        parser.error("--smoke et --check sont exclusifs")
    if a.smoke:
        a.epochs = 2
    if a.stage == "run" and not a.plan:
        parser.error("--plan est obligatoire avec --stage run")
    if a.stage == "status":
        status(a)
    else:
        main_torch(a)
