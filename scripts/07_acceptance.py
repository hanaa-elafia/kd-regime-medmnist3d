"""Regle d'acceptation teacher / student (validation uniquement).

Un teacher est ACCEPTE pour un student si l'intervalle de confiance bootstrap
(95 %) de la difference de NLL de validation (student - teacher) est
entierement au-dessus de zero. Regle ecrite avant les resultats, recommandee
avec au moins 5 seeds.

Lancer (apres 06_screen_models.py) :
    python scripts/07_acceptance.py --dataset fracturemnist3d
Sortie : outputs/screen_<dataset>/acceptance.csv + tableau dans le terminal
"""
import argparse
import glob

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
parser.add_argument("--boot", type=int, default=20000)
args = parser.parse_args()
FLAG = args.dataset.lower()
folder = f"outputs/screen_{FLAG}"

files = sorted(f for f in glob.glob(f"{folder}/results*.csv"))
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df = df.drop_duplicates(subset=["model", "seed"], keep="last")
students = [m for m in ["S1", "S2", "S3"] if m in set(df["model"])]
teachers = sorted(set(df[df["role"] == "teacher"]["model"]))
rng = np.random.default_rng(0)


def boot_diff(a, b, n):
    a, b = np.asarray(a), np.asarray(b)
    d = np.empty(n)
    for i in range(n):
        d[i] = rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean()
    return np.percentile(d, [2.5, 97.5])


print(f"Dataset {FLAG} | fichiers : {[f.split('/')[-1] for f in files]}")
print("\nQualite par modele (validation) :")
summ = (df.groupby("model")[["val_nll", "val_macro_f1", "val_ece", "val_auc"]]
        .agg(["mean", "std"]).round(4))
print(summ.to_string())

rows = []
for s in students:
    sv = df[df["model"] == s].sort_values("seed")["val_nll"].values
    for t in teachers:
        tv = df[df["model"] == t].sort_values("seed")["val_nll"].values
        lo, hi = boot_diff(sv, tv, args.boot)
        f1_gap = (df[df["model"] == t]["val_macro_f1"].mean()
                  - df[df["model"] == s]["val_macro_f1"].mean())
        rows.append({"student": s, "teacher": t, "n_seeds_s": len(sv),
                     "n_seeds_t": len(tv),
                     "nll_gap_mean": sv.mean() - tv.mean(),
                     "ci_low": lo, "ci_high": hi,
                     "f1_gap_mean": f1_gap,
                     "accepted": bool(lo > 0)})
res = pd.DataFrame(rows)
res.to_csv(f"{folder}/acceptance.csv", index=False)

print("\nRegle d'acceptation : IC95 bootstrap de (NLL student - NLL teacher) > 0")
print(res.round(4).to_string(index=False))
n_min = int(res[["n_seeds_s", "n_seeds_t"]].min().min())
if n_min < 5:
    print(f"\nATTENTION : seulement {n_min} seeds par modele. "
          "Resultat indicatif, la regle demande au moins 5 seeds.")
print("\nCouples acceptes :", int(res["accepted"].sum()), "/", len(res))
print("Envoie-moi tout le texte ci-dessus.")
