"""Analyse de la phase B2a, selon addendum_B2a_v1.md (a lancer UNE SEULE FOIS, a la fin de B2a).

Lit outputs/phaseB2a_<dataset>/results*.csv (Student seul et Vanilla, seeds 15 a 24) et ecrit dans
outputs/analysis_b2a/. Aucune metrique de test n'est utilisee pour choisir quoi que ce soit.

ARRETS DE SECURITE (aucune analyse si un controle echoue) : jobs manquants ou hors plan, doublons,
epochs != 80, runs divergents, valeurs de test absentes, seeds differentes de 15 a 24.

Pour chaque condition (dataset, teacher, student) :
  - gain = Macro-F1 de test de Vanilla - Macro-F1 de test du Student seul, par seed (apparie) ;
  - moyenne, ecart-type, IC a 95 % de Student (t, 9 ddl), p-valeur du test t apparie, bootstrap en
    complement ;
  - correction de Holm par dataset, sur les 9 conditions ;
  - classe, dans cet ordre : A (la KD aide : IC > 0 et p ajuste < 0,05), C (la KD nuit : IC < 0 et
    p ajuste < 0,05), B (sans effet : IC dans [-0,02 ; +0,02]), sinon non classe ; drapeau si A et
    IC dans ]0 ; 0,02] (gain negligeable) ;
  - metriques secondaires (balanced accuracy, AUC, ECE, NLL) : gain et IC, descriptifs, sans verdict.
Hypotheses HB1 a HB3 de l'addendum, evaluees avec les criteres ecrits dans l'addendum.
Robustesse descriptive : gain moyen des 7 autres schedules de la Phase B (5 seeds) a cote.
Pas de test global agregeant les 27 conditions : les decomptes sont descriptifs.
"""
import glob
import hashlib
import importlib.util
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


pb = _load("pb", "09_phase_b.py")
an = _load("an", "10_analysis.py")
OUT = Path("outputs/analysis_b2a")
DELTA, ALPHA = an.DELTA, an.ALPHA
PRIMARY, SECONDARY = an.PRIMARY, an.SECONDARY
SEEDS = list(range(15, 25))
CLS = {"gagne": "A", "perd": "C", "egalite": "B", "non_concluant": "-"}


def integrity(ds, res):
    if res.empty:
        return ["aucun resultat"]
    pbs = []
    exp = {pb.job_key(*j) for j in pb.make_jobs("b2a", pb.ALL_TEACHERS, pb.ALL_STUDENTS)}
    done = pb.done_keys(res)
    if exp - done:
        pbs.append(f"{len(exp - done)} job(s) manquant(s), ex. {sorted(exp - done)[:3]}")
    if done - exp:
        pbs.append(f"{len(done - exp)} job(s) hors plan, ex. {sorted(done - exp)[:3]}")
    dups = pb.duplicated_keys(res)
    if dups:
        pbs.append(f"{len(dups)} job(s) en double, ex. {dups[:3]}")
    if (res["epochs"] != pb.EPOCHS).any():
        pbs.append(f"epochs differentes de {pb.EPOCHS}")
    if res["diverged"].astype(bool).any():
        pbs.append("run(s) divergent(s)")
    if not np.isfinite(res["val_nll"]).all():
        pbs.append("NLL de validation non finie")
    if res["test_macro_f1"].isna().any():
        pbs.append("test ABSENT pour des runs")
    if sorted(res["seed"].unique()) != SEEDS:
        pbs.append(f"seeds presentes : {sorted(res['seed'].unique())} (attendu 15 a 24)")
    return pbs


def conditions(ds, res):
    rng = np.random.default_rng(an.BOOT_SEED)
    base = res.copy()
    pivots = {m: base.pivot_table(index=["teacher", "student", "seed"], columns="method",
                                  values=m, aggfunc="mean") for m in [PRIMARY] + SECONDARY}
    rows = []
    for t in pb.ALL_TEACHERS:
        for s in pb.ALL_STUDENTS:
            for metric, piv in pivots.items():
                v = piv.loc[(t, s)]["vanilla"]
                st = piv.loc[("none", s)]["student"]
                d = (v - st).dropna()
                rows.append(dict(dataset=ds, teacher=t, student=s, metric=metric, n=len(d),
                                 **an.paired_stats(d.values, rng)))
    df = pd.DataFrame(rows)
    prim = df["metric"] == PRIMARY
    df["p_holm"] = np.nan
    df.loc[prim, "p_holm"] = an.holm_adjust(df.loc[prim, "p"].values)
    df["verdict"] = ""
    df.loc[prim, "verdict"] = [an.verdict_of(lo, hi, pa) for lo, hi, pa in
                               zip(df.loc[prim, "ci_lo"], df.loc[prim, "ci_hi"],
                                   df.loc[prim, "p_holm"])]
    df["classe"] = df["verdict"].map(CLS).fillna("")
    df["gain_negligeable"] = prim & (df["classe"] == "A") & (df["ci_hi"] <= DELTA)
    return df


def phaseB_reference(ds):
    """Gain moyen (5 seeds) des 7 autres schedules de la Phase B, par condition : descriptif."""
    res = pb.read_results(f"outputs/phaseB_{ds}")
    if res.empty:
        return None
    res = res[(res["method"] != "static_grid") & (res["seed"] < 5)]
    piv = res.pivot_table(index=["teacher", "student", "seed"], columns="method",
                          values=PRIMARY, aggfunc="mean")
    out = []
    others = ["static_best", "lin_T", "lin_A", "lin_TA", "fts", "fas", "dfs"]
    for t in pb.ALL_TEACHERS:
        for s in pb.ALL_STUDENTS:
            kd = piv.loc[(t, s)]
            stu = piv.loc[("none", s)]["student"]
            g = {m: float((kd[m] - stu).mean()) for m in ["vanilla"] + others}
            out.append(dict(dataset=ds, teacher=t, student=s,
                            phaseB_vanilla_gain_5seeds=g["vanilla"],
                            phaseB_autres7_gain_moyen=float(np.mean([g[m] for m in others]))))
    return pd.DataFrame(out)


def hypotheses(cond):
    p = cond[cond["metric"] == PRIMARY]
    rows = []
    f = p[p.dataset == "fracturemnist3d"]
    nA, mean = int((f.classe == "A").sum()), float(f["mean"].mean())
    verdict = ("soutenue" if (nA == 9 and mean > DELTA) else
               "partiellement soutenue" if 6 <= nA <= 8 else "refutee")
    note = "cas limite : 9 conditions de classe A mais gain moyen <= delta" if (nA == 9 and mean <= DELTA) else ""
    rows.append(dict(hypothese="HB1", dataset="fracturemnist3d", verdict=verdict,
                     detail=f"{nA}/9 conditions de classe A ; gain moyen {mean:+.4f} (delta = {DELTA}) {note}".strip()))
    o = p[p.dataset == "organmnist3d"]
    nA, nB, nC = int((o.classe == "A").sum()), int((o.classe == "B").sum()), int((o.classe == "C").sum())
    verdict = ("soutenue" if (nA == 0 and nC == 0 and nB >= 7) else
               "refutee" if (nA >= 1 or nC >= 1) else "partiellement soutenue")
    rows.append(dict(hypothese="HB2", dataset="organmnist3d", verdict=verdict,
                     detail=f"classe A : {nA}, B : {nB}, C : {nC}, non classees : {9 - nA - nB - nC}"))
    a = p[p.dataset == "adrenalmnist3d"]
    s1 = a[a.student == "S1"]
    other = a[a.student != "S1"]
    cond_s1 = int((s1.classe == "A").sum())
    cond_o = int((other.classe == "A").sum())
    verdict = "soutenue" if (cond_s1 == 3 and cond_o == 0) else "refutee"
    rows.append(dict(hypothese="HB3", dataset="adrenalmnist3d", verdict=verdict,
                     detail=f"S1 : {cond_s1}/3 de classe A ; S2 et S3 : {cond_o}/6 de classe A"))
    return pd.DataFrame(rows)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    datasets = pb.DATASETS
    allres, bad = {}, {}
    for ds in datasets:
        allres[ds] = pb.read_results(f"outputs/phaseB2a_{ds}")
        bad[ds] = integrity(ds, allres[ds])
    if any(bad.values()):
        print("ARRET : controles d'integrite en echec. Aucune analyse n'est faite.\n")
        for ds, p in bad.items():
            if p:
                print(f"[{ds}]")
                for x in p:
                    print("  -", x)
        raise SystemExit(1)
    print("Controles d'integrite : OK pour", ", ".join(datasets), "\n")
    OUT.mkdir(parents=True, exist_ok=True)
    cond = pd.concat([conditions(ds, allres[ds]) for ds in datasets], ignore_index=True)
    refs = [phaseB_reference(ds) for ds in datasets]
    refs = pd.concat([r for r in refs if r is not None], ignore_index=True) if any(r is not None for r in refs) else None
    if refs is not None:
        prim = cond[cond["metric"] == PRIMARY].merge(refs, on=["dataset", "teacher", "student"], how="left")
        cond = pd.concat([prim, cond[cond["metric"] != PRIMARY]], ignore_index=True)
    cond.to_csv(OUT / "b2a_conditions.csv", index=False)
    hyp = hypotheses(cond)
    hyp.to_csv(OUT / "b2a_hypotheses.csv", index=False)
    with open(OUT / "analysis_info.txt", "w", encoding="utf-8") as fh:
        fh.write(f"Analyse B2a lancee le {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        fh.write(f"delta={DELTA} alpha={ALPHA} seeds={SEEDS[0]}-{SEEDS[-1]} holm=par dataset sur 9 conditions\n\n")
        for ds in datasets:
            for fp in sorted(glob.glob(f"outputs/phaseB2a_{ds}/results*.csv")):
                fh.write(f"{sha256(fp)}  {fp}\n")
    pd.set_option("display.width", 220)
    p = cond[cond["metric"] == PRIMARY].copy()
    print("=== Carte des 27 conditions : classe (A = la KD aide, B = sans effet, C = la KD nuit, - = non classe) et gain de F1 ===")
    print("(* = gain etabli mais inferieur a delta = 0,02)\n")
    for ds in datasets:
        q = p[p.dataset == ds].copy()
        q["txt"] = [f"{c}{'*' if n else ''} {m:+.3f}" for c, n, m in zip(q.classe, q.gain_negligeable, q["mean"])]
        print(f"--- {ds} ---")
        print(q.pivot(index="teacher", columns="student", values="txt").to_string(), "\n")
    print("=== Decompte descriptif des classes ===")
    print(p.groupby(["dataset", "classe"]).size().unstack(fill_value=0).to_string(), "\n")
    print("=== Hypotheses de l'addendum (criteres ecrits avant le calcul) ===")
    print(hyp.to_string(index=False))
    print(f"\nFichiers ecrits dans {OUT}/ : b2a_conditions.csv, b2a_hypotheses.csv, analysis_info.txt")


if __name__ == "__main__":
    main()
