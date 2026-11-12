import pandas as pd, numpy as np
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50); pd.set_option("display.max_rows", 500)
c = pd.read_csv("outputs/analysis/comparisons_long.csv")
METRICS = ["test_macro_f1", "test_bal_acc", "test_auc", "test_ece", "test_nll"]
LOWER = {"test_ece", "test_nll"}
SHORT = {"test_macro_f1": "F1", "test_bal_acc": "BalAcc", "test_auc": "AUC", "test_ece": "ECE", "test_nll": "NLL"}
NAMES = {"C:fts-fas": "FTS contre FAS", "D1:dfs-fts": "DFS contre FTS", "D2:dfs-fas": "DFS contre FAS"}
c = c[c.comparison.isin(NAMES)].copy().reset_index(drop=True)
sgn = np.where(c.metric.isin(LOWER), -1, 1)
c["gain"] = c["mean"] * sgn                                   # > 0 : la 1re methode est MEILLEURE que la 2e
c["lo"] = np.where(sgn == 1, c.ci_lo, -c.ci_hi); c["hi"] = np.where(sgn == 1, c.ci_hi, -c.ci_lo)
c["res"] = np.where(c.lo > 0, "1re MIEUX", np.where(c.hi < 0, "1re PIRE", "="))
o = {m: i for i, m in enumerate(METRICS)}
print("=" * 100)
print("PARTIE 1 - PIVOT (15 seeds) : une methode fuzzy contre une autre. gain > 0 = la PREMIERE est meilleure (signe deja inverse pour ECE et NLL)")
print("=" * 100)
p = c[c.scope.str.startswith("confirmatoire")]
for ds in ["fracturemnist3d", "adrenalmnist3d", "organmnist3d"]:
    q = p[p.dataset == ds]
    if q.empty: print(f"\n--- {ds} : pas de donnees ---"); continue
    q = q.assign(m=q.metric.map(o)).sort_values(["comparison", "m"])
    print(f"\n--- {ds} ---")
    print(pd.DataFrame([dict(comparaison=NAMES[r.comparison], metrique=SHORT[r.metric], gain=f"{r.gain:+.4f}",
                             ic95=f"[{r.lo:+.4f}, {r.hi:+.4f}]", resultat=r.res) for r in q.itertuples()]).to_string(index=False))
print("\n" + "=" * 100)
print("PARTIE 2 - 9 CONDITIONS, 5 seeds, EXPLORATOIRE : gain moyen (nombre de conditions ou la 1re methode est meilleure)")
print("=" * 100)
e = c[c.scope.str.startswith("exploratoire")]
g = e.groupby(["dataset", "comparison", "metric"]).agg(moy=("gain", "mean"), k=("gain", lambda s: int((s > 0).sum()))).reset_index()
g["txt"] = g.apply(lambda r: f"{r.moy:+.4f} ({r.k}/9)", axis=1)
for ds in ["fracturemnist3d", "adrenalmnist3d", "organmnist3d"]:
    if not (g.dataset == ds).any(): print(f"\n--- {ds} : pas de donnees ---"); continue
    t = g[g.dataset == ds].pivot(index="comparison", columns="metric", values="txt")[METRICS].rename(columns=SHORT)
    t.index = [NAMES[i] for i in t.index]
    print(f"\n--- {ds} ---"); print(t.to_string())
print("\nRappel : le F1 est la metrique primaire (verdicts de la fiche) ; les autres sont descriptives.")
