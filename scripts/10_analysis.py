"""Analyse finale de la Phase B, selon la fiche de gel (sections 8 et 9).

Deux etapes :
  --stage teachers_test   (GPU/torch) evalue UNE FOIS chaque teacher sur le test, depuis son
                          checkpoint, apres avoir verifie que l'evaluation est correcte :
                          la NLL de VALIDATION recalculee doit egaler celle enregistree
                          (pour chaque teacher et pour 5 checkpoints de grille par dataset).
  --stage analyze         (numpy/scipy) tableaux, intervalles, verdicts, variables de regime.

Ce que fait `analyze` (rien n'est choisi sur le test) :
  - ARRETS DE SECURITE : jobs manquants, doublons, epochs != 80, runs divergents, test present
    pour la grille Static ou absent ailleurs, point Static non reproductible depuis la
    validation. Si un controle echoue, aucune analyse n'est faite.
  - Gain d'une methode M contre le Student = metrique(M) - metrique(Student), par seed.
  - IC a 95 % de Student (t, n-1 ddl) sur les differences appariees ; IC bootstrap
    (percentile, 10 000 reechantillonnages, graine fixe) en complement.
  - Metrique primaire : Macro-F1 de test. Secondaires (descriptives, sans verdict) :
    Balanced Accuracy, AUC, ECE, NLL de test.
  - Comparaisons cles (14 par couple) : G = FTS, FAS, DFS contre Student ; A-D = A fts-vanilla,
    B fas-vanilla, C fts-fas, D1 dfs-fts, D2 dfs-fas ; E = fts-lin_T, fas-lin_A, dfs-lin_TA ;
    F = fts, fas, dfs contre Static au meilleur point.
  - Correction de Holm sur chaque famille, par dataset (et par portee : exploratoire sur les
    9 couples avec 5 seeds ; confirmatoire sur le couple pivot avec 15 seeds).
  - Verdicts, dans l'ordre de la fiche : gagne (IC > 0 et p ajuste < 0,05), perd (IC < 0 et
    p ajuste < 0,05), egalite (IC dans [-0,02 ; +0,02]), sinon non_concluant. Une colonne
    signale les cas ou l'IC est a la fois au-dessus de 0 et dans [0 ; 0,02] (gain negligeable).
  - AUCUNE agregation des couples en un test unique : les decomptes par verdict sont
    descriptifs.

Interpretations declarees (la fiche est ambigue sur ces points ; a noter dans l'article) :
  (i) l'ablation D (FTS + FAS contre DFS) est operationnalisee par deux comparaisons, dfs-fts et
      dfs-fas ; (ii) la famille de Holm de la portee exploratoire regroupe tous les couples d'un
      dataset ; (iii) en cas de chevauchement gagne / egalite, l'ordre de la fiche prevaut.

Sorties dans outputs/analysis/ : comparisons_long.csv, verdicts_primary.csv, verdict_counts.csv,
descriptives.csv, regime_variables.csv, teachers_test.csv (si l'etape teachers_test a ete faite),
analysis_info.txt (parametres et empreintes des fichiers de resultats).

Lancement :
  python scripts/10_analysis.py --stage teachers_test        (une fois, avant analyze)
  python scripts/10_analysis.py --stage analyze              (une fois, a la fin)
"""
import argparse
import glob
import hashlib
import importlib.util
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("pb", HERE / "09_phase_b.py")
pb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pb)

OUT = Path("outputs/analysis")
DELTA, ALPHA, N_BOOT, BOOT_SEED = 0.02, 0.05, 10000, 12345
PRIMARY = "test_macro_f1"
SECONDARY = ["test_bal_acc", "test_auc", "test_ece", "test_nll"]
VERDICT_FAMILIES = ["vs_student", "A-D", "E", "F"]
COMPARISONS = [
    ("vs_student", "fts-student", "fts", "student"),
    ("vs_student", "fas-student", "fas", "student"),
    ("vs_student", "dfs-student", "dfs", "student"),
    ("A-D", "A:fts-vanilla", "fts", "vanilla"),
    ("A-D", "B:fas-vanilla", "fas", "vanilla"),
    ("A-D", "C:fts-fas", "fts", "fas"),
    ("A-D", "D1:dfs-fts", "dfs", "fts"),
    ("A-D", "D2:dfs-fas", "dfs", "fas"),
    ("E", "E:fts-lin_T", "fts", "lin_T"),
    ("E", "E:fas-lin_A", "fas", "lin_A"),
    ("E", "E:dfs-lin_TA", "dfs", "lin_TA"),
    ("F", "F:fts-static", "fts", "static_best"),
    ("F", "F:fas-static", "fas", "static_best"),
    ("F", "F:dfs-static", "dfs", "static_best"),
]
DESCRIPTIVE = [
    ("desc", "vanilla-student", "vanilla", "student"),
    ("desc", "static-student", "static_best", "student"),
    ("desc", "lin_T-student", "lin_T", "student"),
    ("desc", "lin_A-student", "lin_A", "student"),
    ("desc", "lin_TA-student", "lin_TA", "student"),
]


# ------------------------------------------------------------- statistiques
def holm_adjust(p):
    p = np.asarray(p, dtype=float)
    adj = np.full_like(p, np.nan)
    m = int(np.sum(~np.isnan(p)))
    order = np.argsort(np.where(np.isnan(p), np.inf, p), kind="stable")
    running, k = 0.0, 0
    for idx in order:
        if np.isnan(p[idx]):
            continue
        running = max(running, (m - k) * p[idx])
        adj[idx] = min(1.0, running)
        k += 1
    return adj


def paired_stats(d, rng):
    d = np.asarray(d, dtype=float)
    n = len(d)
    out = dict(mean=np.nan, sd=np.nan, ci_lo=np.nan, ci_hi=np.nan, p=np.nan,
               boot_lo=np.nan, boot_hi=np.nan)
    if n < 2:
        out["mean"] = float(d.mean()) if n else np.nan
        return out
    mean, sd = float(d.mean()), float(d.std(ddof=1))
    out["mean"], out["sd"] = mean, sd
    if sd == 0.0:   # differences toutes identiques (ex. F1 sature) : intervalle degenere
        out.update(ci_lo=mean, ci_hi=mean, p=1.0 if mean == 0.0 else 0.0)
    else:
        half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
        out.update(ci_lo=mean - half, ci_hi=mean + half,
                   p=float(stats.ttest_1samp(d, 0.0).pvalue))
    boot = rng.choice(d, size=(N_BOOT, n), replace=True).mean(axis=1)
    out["boot_lo"], out["boot_hi"] = [float(x) for x in np.percentile(boot, [2.5, 97.5])]
    return out


def verdict_of(lo, hi, p_adj):
    if np.isnan(lo) or np.isnan(p_adj):
        return "non_concluant"
    if lo > 0 and p_adj < ALPHA:
        return "gagne"
    if hi < 0 and p_adj < ALPHA:
        return "perd"
    if lo >= -DELTA and hi <= DELTA:
        return "egalite"
    return "non_concluant"


# ----------------------------------------------------------- controles
def integrity(ds, res):
    """Retourne la liste des problemes (vide = OK). Aucun controle ne lit une valeur de test,
    sauf la presence ou l'absence."""
    if res.empty:
        return ["aucun resultat"]
    pbs = []
    done = pb.done_keys(res)
    exp = set()
    for plan in ("students", "primary", "pivot"):
        exp |= {pb.job_key(*j) for j in pb.make_jobs(plan, pb.ALL_TEACHERS, pb.ALL_STUDENTS)}
    exp |= {pb.job_key(t, s, "static_best", "", sd) for t in pb.ALL_TEACHERS
            for s in pb.ALL_STUDENTS for sd in range(3)}
    miss = exp - done
    if miss:
        pbs.append(f"{len(miss)} job(s) manquant(s), ex. {sorted(miss)[:3]}")
    extra = done - exp
    if extra:
        pbs.append(f"{len(extra)} job(s) hors plan, ex. {sorted(extra)[:3]}")
    dups = pb.duplicated_keys(res)
    if dups:
        pbs.append(f"{len(dups)} job(s) en double, ex. {dups[:3]}")
    if (res["epochs"] != pb.EPOCHS).any():
        pbs.append(f"epochs differentes de {pb.EPOCHS} : {sorted(res['epochs'].unique())}")
    if res["diverged"].astype(bool).any():
        pbs.append("run(s) divergent(s)")
    if not np.isfinite(res["val_nll"]).all():
        pbs.append("NLL de validation non finie")
    grid = res[res["method"] == "static_grid"]
    if grid["test_macro_f1"].notna().any():
        pbs.append("test PRESENT pour la grille Static (ne doit pas l'etre)")
    other = res[res["method"] != "static_grid"]
    if other["test_macro_f1"].isna().any():
        pbs.append("test ABSENT pour des runs retenus")
    sb = res[res["method"] == "static_best"]
    if len(sb) and (sb.groupby(["teacher", "student"])["config"].nunique() != 1).any():
        pbs.append("point Static non unique pour un couple")
    if not pbs:   # reproductibilite du choix Static depuis la validation seule
        for (t, s), g in sb.groupby(["teacher", "student"]):
            if pb.resolve_best(res, t, s) != g["config"].iloc[0]:
                pbs.append(f"point Static de {t} x {s} non reproductible depuis la validation")
    return pbs


# --------------------------------------------------------- tables
def couple_frame(piv, t, s):
    kd = piv.loc[(t, s)].drop(columns=["student"], errors="ignore")
    stu = piv.loc[("none", s)]["student"].rename("student")
    return kd.join(stu, how="inner")


def run_scope(ds, res, scope, couples, seeds):
    rng = np.random.default_rng(BOOT_SEED)
    base = res[res["method"] != "static_grid"]
    pivots = {m: base.pivot_table(index=["teacher", "student", "seed"], columns="method",
                                  values=m, aggfunc="mean")
              for m in [PRIMARY] + SECONDARY}
    rows = []
    for t, s in couples:
        for metric, piv in pivots.items():
            cf = couple_frame(piv, t, s)
            cf = cf[cf.index.isin(seeds)]
            for fam, name, a, b in COMPARISONS + DESCRIPTIVE:
                d = (cf[a] - cf[b]).dropna()
                rows.append(dict(dataset=ds, scope=scope, teacher=t, student=s, family=fam,
                                 comparison=name, metric=metric, n=len(d),
                                 **paired_stats(d.values, rng)))
    df = pd.DataFrame(rows)
    df["p_holm"] = np.nan
    df["verdict"] = ""
    prim = (df["metric"] == PRIMARY) & df["family"].isin(VERDICT_FAMILIES)
    for fam in VERDICT_FAMILIES:
        idx = df.index[prim & (df["family"] == fam)]
        df.loc[idx, "p_holm"] = holm_adjust(df.loc[idx, "p"].values)
    df.loc[prim, "verdict"] = [verdict_of(lo, hi, pa) for lo, hi, pa in
                               zip(df.loc[prim, "ci_lo"], df.loc[prim, "ci_hi"],
                                   df.loc[prim, "p_holm"])]
    df["ic_dans_0_delta"] = (df["ci_lo"] > 0) & (df["ci_hi"] <= DELTA)
    df.loc[df["family"] == "desc", "verdict"] = "descriptif"
    return df


def descriptives(ds, res):
    cols = [PRIMARY] + SECONDARY + ["val_nll", "val_macro_f1"]
    base = res[res["method"] != "static_grid"].copy()
    out = []
    for scope, seeds, sel in (("exploratoire", range(5), None),
                              ("pivot_15_seeds", range(15), (pb.PIVOT_TEACHER, pb.PIVOT_STUDENT))):
        b = base[base["seed"].isin(list(seeds))]
        if sel:
            b = b[((b["teacher"] == sel[0]) | (b["teacher"] == "none")) & (b["student"] == sel[1])]
        g = b.groupby(["teacher", "student", "method"])[cols].agg(["mean", "std", "count"])
        g.columns = [f"{c}_{a}" for c, a in g.columns]
        g = g.reset_index()
        g.insert(0, "scope", scope)
        g.insert(0, "dataset", ds)
        out.append(g)
    return pd.concat(out, ignore_index=True)


def regime_variables(ds, res, tt):
    d = f"outputs/phaseB_{ds}"
    tr = pd.read_csv(f"{d}/teachers_results.csv").set_index("teacher")
    ep_files = sorted(glob.glob(f"{d}/epochs*.csv"))
    gaps = None
    if ep_files:
        ep = pd.concat([pd.read_csv(f, usecols=["epoch", "gap_kl_t1", "teacher", "student",
                                                "method", "seed"]) for f in ep_files])
        ep = ep[(ep["method"] == "vanilla") & (ep["epoch"] == pb.EPOCHS)]
        gaps = ep.groupby(["teacher", "student"])["gap_kl_t1"].mean()
    stu = res[(res["method"] == "student") & (res["seed"] < 5)].groupby("student")
    rows = []
    for t in pb.ALL_TEACHERS:
        for s in pb.ALL_STUDENTS:
            sub = res[(res["teacher"] == t) & (res["student"] == s)].iloc[0]
            r = dict(dataset=ds, teacher=t, student=s,
                     param_ratio=sub["teacher_params"] / sub["student_params"],
                     flops_ratio=sub["teacher_mflops"] / sub["student_mflops"],
                     teacher_val_nll=tr.loc[t, "val_nll"], teacher_val_f1=tr.loc[t, "val_macro_f1"],
                     teacher_val_ece=tr.loc[t, "val_ece"], teacher_val_auc=tr.loc[t, "val_auc"],
                     student_val_nll_mean=stu["val_nll"].mean()[s],
                     student_test_f1_mean=stu[PRIMARY].mean()[s],
                     residual_gap_vanilla_epoch80=(gaps.get((t, s), np.nan)
                                                   if gaps is not None else np.nan))
            if tt is not None:
                x = tt[(tt["dataset"] == ds) & (tt["teacher"] == t)]
                if len(x):
                    r.update(teacher_test_f1=x["test_macro_f1"].iloc[0],
                             teacher_test_ece=x["test_ece"].iloc[0],
                             teacher_test_auc=x["test_auc"].iloc[0],
                             teacher_test_nll=x["test_nll"].iloc[0])
            rows.append(r)
    return pd.DataFrame(rows)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stage_analyze(args):
    OUT.mkdir(parents=True, exist_ok=True)
    all_res, problems = {}, {}
    for ds in args.datasets:
        res = pb.read_results(f"outputs/phaseB_{ds}")
        all_res[ds] = res
        problems[ds] = integrity(ds, res)
    bad = {ds: p for ds, p in problems.items() if p}
    if bad:
        print("ARRET : controles d'integrite en echec. Aucune analyse n'est faite.\n")
        for ds, p in bad.items():
            print(f"[{ds}]")
            for x in p:
                print("  -", x)
        raise SystemExit(1)
    print("Controles d'integrite : OK pour", ", ".join(args.datasets), "\n")

    tt_path = OUT / "teachers_test.csv"
    tt = pd.read_csv(tt_path) if tt_path.exists() else None
    if tt is None:
        print("Note : teachers_test.csv absent (etape teachers_test non faite) : "
              "les variables de regime n'auront pas les metriques de test des teachers.\n")
    couples = [(t, s) for t in pb.ALL_TEACHERS for s in pb.ALL_STUDENTS]
    longs, descs, regs = [], [], []
    for ds in args.datasets:
        res = all_res[ds]
        longs.append(run_scope(ds, res, "exploratoire_5_seeds", couples, set(range(5))))
        longs.append(run_scope(ds, res, "confirmatoire_pivot_15_seeds",
                               [(pb.PIVOT_TEACHER, pb.PIVOT_STUDENT)], set(range(15))))
        descs.append(descriptives(ds, res))
        regs.append(regime_variables(ds, res, tt))
    long = pd.concat(longs, ignore_index=True)
    long.to_csv(OUT / "comparisons_long.csv", index=False)
    verd = long[(long["metric"] == PRIMARY) & long["family"].isin(VERDICT_FAMILIES)].copy()
    verd.to_csv(OUT / "verdicts_primary.csv", index=False)
    counts = (verd.groupby(["dataset", "scope", "family", "verdict"]).size()
              .unstack("verdict", fill_value=0).reset_index())
    counts.to_csv(OUT / "verdict_counts.csv", index=False)
    pd.concat(descs, ignore_index=True).to_csv(OUT / "descriptives.csv", index=False)
    pd.concat(regs, ignore_index=True).to_csv(OUT / "regime_variables.csv", index=False)
    with open(OUT / "analysis_info.txt", "w", encoding="utf-8") as f:
        f.write(f"Analyse lancee le {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"delta={DELTA} alpha={ALPHA} n_boot={N_BOOT} boot_seed={BOOT_SEED}\n")
        f.write(f"metrique primaire={PRIMARY} ; secondaires={SECONDARY}\n")
        f.write(f"comparaisons cles={[c[1] for c in COMPARISONS]}\n\nEmpreintes SHA-256 :\n")
        for ds in args.datasets:
            for fp in sorted(glob.glob(f"outputs/phaseB_{ds}/results*.csv")):
                f.write(f"{sha256(fp)}  {fp}\n")
    pd.set_option("display.width", 200)
    print("=== Decompte des verdicts (Macro-F1 de test ; DESCRIPTIF, pas un test global) ===")
    print(counts.to_string(index=False))
    print(f"\nFichiers ecrits dans {OUT}/ : comparisons_long.csv, verdicts_primary.csv, "
          "verdict_counts.csv, descriptives.csv, regime_variables.csv, analysis_info.txt")


# ------------------------------------------------ teachers sur le test (torch)
def stage_teachers_test(args):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import medmnist
    from medmnist import INFO
    from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
                "ece": ece_score(probs, y_t), "auc": auc_score(probs.numpy(), y_np, n_cls)}

    OUT.mkdir(parents=True, exist_ok=True)
    rows, rng = [], np.random.default_rng(0)
    for ds in args.datasets:
        d = f"outputs/phaseB_{ds}"
        info = INFO[ds]
        DataClass = getattr(medmnist, info["python_class"])
        n_cls = len(info["label"])

        def load(split):
            m = DataClass(split=split, download=True, size=28)
            x = torch.from_numpy(m.imgs).float().div(255.0)
            x = x.unsqueeze(1) if x.ndim == 4 else x
            return x, torch.from_numpy(m.labels.flatten()).long()

        (xtr, _), (xva, yva), (xte, yte) = load("train"), load("val"), load("test")
        mean, std = xtr.mean().item(), xtr.std().item()
        xva, xte = (xva - mean) / std, (xte - mean) / std
        models = build_models(n_cls)
        tr = pd.read_csv(f"{d}/teachers_results.csv").set_index("teacher")
        res = pb.read_results(d)
        # --- verification de l'evaluation (validation seulement)
        for t in pb.ALL_TEACHERS:
            m = models[t]().to(dev)
            assert sum(p.numel() for p in m.parameters()) == int(tr.loc[t, "params"]), t
            m.load_state_dict(torch.load(f"{d}/teacher_{t}.pt", map_location=dev))
            v = F.cross_entropy(logits_of(m, xva), yva).item()
            if abs(v - float(tr.loc[t, "val_nll"])) > 1e-4:
                raise SystemExit(f"ARRET : NLL de validation recalculee ({v:.6f}) differente de "
                                 f"celle enregistree ({tr.loc[t, 'val_nll']:.6f}) pour {ds} {t}")
            print(f"[{ds}] {t} : verification OK (NLL de validation recalculee = {v:.4f})")
        grid = res[res["method"] == "static_grid"]
        for _, r in grid.sample(n=min(5, len(grid)), random_state=int(rng.integers(1_000_000))).iterrows():
            m = models[r["student"]]().to(dev)
            m.load_state_dict(torch.load(f"{d}/ckpt/{r['teacher']}_{r['student']}_{r['config']}_s{int(r['seed'])}.pt",
                                         map_location=dev))
            v = F.cross_entropy(logits_of(m, xva), yva).item()
            if abs(v - float(r["val_nll"])) > 1e-4:
                raise SystemExit(f"ARRET : checkpoint de grille incoherent ({ds} {r['teacher']} "
                                 f"{r['student']} {r['config']} seed {int(r['seed'])})")
        print(f"[{ds}] 5 checkpoints de grille : verification OK")
        # --- evaluation unique sur le test
        for t in pb.ALL_TEACHERS:
            m = models[t]().to(dev)
            m.load_state_dict(torch.load(f"{d}/teacher_{t}.pt", map_location=dev))
            te = evaluate(m, xte, yte, yte.numpy(), n_cls)
            rows.append(dict(dataset=ds, teacher=t, test_macro_f1=te["f1"],
                             test_bal_acc=te["bacc"], test_ece=te["ece"],
                             test_auc=te["auc"], test_nll=te["nll"]))
    pd.DataFrame(rows).to_csv(OUT / "teachers_test.csv", index=False)
    print(f"\nTeachers evalues une seule fois sur le test : {OUT / 'teachers_test.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=["analyze", "teachers_test"])
    parser.add_argument("--datasets", nargs="+", default=pb.DATASETS, choices=pb.DATASETS)
    a = parser.parse_args()
    if a.stage == "analyze":
        stage_analyze(a)
    else:
        stage_teachers_test(a)
