import pandas as pd, numpy as np
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50); pd.set_option("display.max_rows", 500)
c = pd.read_csv("outputs/analysis/comparisons_long.csv")
METRICS = ["test_macro_f1", "test_bal_acc", "test_auc", "test_ece", "test_nll"]
LOWER = {"test_ece", "test_nll"}                      # pour ces deux : plus bas = mieux
SHORT = {"test_macro_f1": "F1", "test_bal_acc": "BalAcc", "test_auc": "AUC", "test_ece": "ECE", "test_nll": "NLL"}
METHODS = ["vanilla", "fts", "fas", "dfs"]            # KD normale + les 3 fuzzy
c = c[c.comparison.str.endswith("-student") & c.comparison.str.replace("-student", "", regex=False).isin(METHODS)].copy().reset_index(drop=True)
c["method"] = c.comparison.str.replace("-student", "", regex=False)
c["bon_sens"] = np.where(c.metric.isin(LOWER), -1, 1)
c["gain_kd"] = c["mean"] * c["bon_sens"]              # >0 = la KD est meilleure, pour toutes les metriques
c["lo_kd"] = np.where(c.bon_sens == 1, c.ci_lo, -c.ci_hi)   # bornes de l'IC du gain "KD meilleure"
c["hi_kd"] = np.where(c.bon_sens == 1, c.ci_hi, -c.ci_lo)
c["verdict"] = np.where(c.lo_kd > 0, "KD MIEUX", np.where(c.hi_kd < 0, "KD PIRE", "="))
order = {m: i for i, m in enumerate(METRICS)}

print("="*100)
print("PARTIE 1 - COUPLE PIVOT (res_w32_b2 x S1), 15 seeds : gain de la KD contre le Student seul")
print("gain positif = la KD est meilleure (pour ECE et NLL, le signe est deja inverse). IC 95 % de Student.")
print("="*100)
p = c[c.scope.str.startswith("confirmatoire")]
for ds in ["fracturemnist3d", "adrenalmnist3d", "organmnist3d"]:
    q = p[p.dataset == ds].copy()
    if q.empty:
        print(f"\n--- {ds} : pas de donnees ---"); continue
    q["o"] = q.metric.map(order); q = q.sort_values(["method", "o"])
    rows = []
    for r in q.itertuples():
        rows.append(dict(methode=r.method, metrique=SHORT[r.metric], gain_kd=f"{r.gain_kd:+.4f}",
                         ic95=f"[{r.lo_kd:+.4f}, {r.hi_kd:+.4f}]", resultat=r.verdict))
    print(f"\n--- {ds} ---"); print(pd.DataFrame(rows).to_string(index=False))

print("\n" + "="*100)
print("RESUME PARTIE 1 - Cas ou la KD est MEILLEURE que le Student seul (l'IC exclut 0), couple pivot")
print("="*100)
w = p[p.verdict == "KD MIEUX"].copy(); w["o"] = w.metric.map(order)
for ds in ["fracturemnist3d", "adrenalmnist3d", "organmnist3d"]:
    x = w[w.dataset == ds].sort_values(["o", "method"])
    if len(x) == 0:
        print(f"{ds:16}: aucun cas"); continue
    print(f"{ds:16}: " + " ; ".join(f"{SHORT[m]}: " + "/".join(x[x.metric == m].method) for m in METRICS if (x.metric == m).any()))
bad = p[p.verdict == "KD PIRE"].copy()
print("\nCas ou la KD est PIRE (l'IC exclut 0) :")
for ds in ["fracturemnist3d", "adrenalmnist3d", "organmnist3d"]:
    x = bad[bad.dataset == ds].copy(); x["o"] = x.metric.map(order); x = x.sort_values(["o", "method"])
    print(f"{ds:16}: " + (" ; ".join(f"{SHORT[m]}: " + "/".join(x[x.metric == m].method) for m in METRICS if (x.metric == m).any()) if len(x) else "aucun cas"))

print("\n" + "="*100)
print("PARTIE 2 - 9 CONDITIONS (3 teachers x 3 students), 5 seeds, EXPLORATOIRE : gain moyen de la KD (et nb de conditions ou la KD est meilleure)")
print("="*100)
e = c[c.scope.str.startswith("exploratoire")]
g = e.groupby(["dataset", "method", "metric"]).agg(moy=("gain_kd", "mean"), k=("gain_kd", lambda s: int((s > 0).sum()))).reset_index()
g["txt"] = g.apply(lambda r: f"{r.moy:+.4f} ({r.k}/9)", axis=1)
for ds in ["fracturemnist3d", "adrenalmnist3d", "organmnist3d"]:
    if not (g.dataset == ds).any():
        print(f"\n--- {ds} : pas de donnees ---"); continue
    t = g[g.dataset == ds].pivot(index="method", columns="metric", values="txt")[METRICS].rename(columns=SHORT)
    print(f"\n--- {ds} ---"); print(t.loc[METHODS].to_string())
print("\nRappel : metriques secondaires (BalAcc, AUC, ECE, NLL) = descriptives, sans correction de comparaisons multiples ; le F1 est la metrique primaire.")
