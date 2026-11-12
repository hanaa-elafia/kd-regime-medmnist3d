# When does knowledge distillation help small 3D medical image classifiers?

Code and frozen protocol for the study *"When does knowledge distillation help small 3D medical image classifiers? A confirmatory study on three MedMNIST datasets"*.

We study knowledge distillation (KD) on three MedMNIST 3D datasets (OrganMNIST3D, AdrenalMNIST3D, FractureMNIST3D; 28x28x28 voxels; official splits), with 3 teachers (3D ResNets), 3 students (small 3D CNNs) and 27 teacher-student-dataset conditions. We compare several ways to set the temperature T and the loss weight alpha: fixed (vanilla, tuned static), linear in time, and a fuzzy controller (FTS, FAS, DFS). The study does not propose a new method. It maps where KD is useful, with a frozen protocol, a protected test set, paired seeds, confidence intervals, Holm correction, an equivalence margin and a confirmatory phase with new seeds (Phase B2a).

> Status: manuscript submitted. Paper reference: **[to be added]**.

## Repository content

| Path | Content |
|---|---|
| `scripts/01_*.py` to `scripts/07_acceptance.py` | Phase A: dataset analysis, resolution probe, capacity sweep, teacher and student screening, acceptance checks (validation data only) |
| `scripts/08_pilot_kd.py` | Pilot of the KD schedules (validation data only) |
| `scripts/09_phase_b.py` | Phase B (main experiments) and Phase B2a (confirmation): teachers, runs, status |
| `scripts/09_phase_b_ancienne_phaseB.py` | Version of `09_phase_b.py` used for Phase B, kept for traceability (before the `b2a` plan was added) |
| `scripts/10_analysis.py` | Analysis of Phase B (run once) |
| `scripts/11_kd_wins.py`, `scripts/12_fuzzy_compare.py` | Read-only summaries of Phase B results (exploratory) |
| `scripts/13_b2a_analysis.py` | Analysis of Phase B2a (run once, rules fixed in the addendum) |
| `scripts/14_env_check.py` | Environment check (validation data only) |
| `scripts/fig_protocol.py`, `scripts/fig_regime_map.py` | Figures 1 and 2 of the paper |
| `scripts/run_phaseB_gpu0.sh`, `run_phaseB_gpu1.sh`, `run_b2a_gpu0.sh`, `run_b2a_gpu1.sh` | Launchers for two GPUs |
| `fiche_gel_protocole_v1.md` | Frozen protocol, in French (hypotheses, bounds, statistics, verdict rules, register of deviations); English translation in Supplementary File S1 of the paper |
| `addendum_B2a_v1.md` | Addendum for Phase B2a, in French (hypotheses HB1 to HB3, classification rules); English translation in Supplementary File S2 |
| `environment.yml` | Conda environment used for the experiments |

The repository also contains the run-level result files (`outputs/`). The nine trained teachers (3 datasets x 3 teachers, files `<dataset>_<teacher>.pt`, with SHA-256 hashes in `sha256.txt`) are attached to the GitHub release **v1.0** of this repository (Releases section). To reuse them, copy each file to `outputs/phaseB_<dataset>/teacher_<teacher>.pt`. Not included: the MedMNIST data, downloaded automatically by the `medmnist` package on first use.

## Where each result of the paper comes from

| Paper | File | Script |
|---|---|---|
| Tables 5 and 6, Figure 2 (Phase B, grey points) | `outputs/phaseB_<dataset>/results*.csv`, `outputs/analysis/` | `10_analysis.py`, `fig_regime_map.py` |
| Table 8 (pivot) and Section 3.7 (schedules) | `outputs/analysis/` | `10_analysis.py`, `12_fuzzy_compare.py` |
| Table 7 and Figure 2 (Phase B2a, coloured points), hypotheses HB1 to HB3 | `outputs/analysis_b2a/b2a_conditions.csv`, `b2a_hypotheses.csv` | `13_b2a_analysis.py`, `fig_regime_map.py` |
| Final temperature T (Section 3.7) | `outputs/phaseB_<dataset>/epochs*.csv` (column `T`, last epoch) | see below |
| Integrity of the B2a result files | `outputs/b2a_resultats_sha256.txt`, `outputs/b2a_versions.txt` | `sha256sum -c` |

[TO CHECK: table numbers match the final PDF]

## Environment

```bash
conda env create -f environment.yml
conda activate kd-medmnist
```

Experiments ran with Python 3.11.16, PyTorch 2.14.1 (CUDA 13.0 build), NumPy 2.4.6, pandas 3.0.6, SciPy 1.17.1, scikit-learn 1.9.1 and MedMNIST 3.0.2, on two NVIDIA RTX A4000 GPUs. Phases B and B2a used the same machine and the same Python environment. GPU determinism was not forced (see the paper), so repeated runs differ slightly.

## How to run

The scripts expect the working directory to contain `scripts/` and `outputs/`.

1. Check the environment (needs the trained teachers in `outputs/phaseB_<dataset>/`):
   ```bash
   python scripts/14_env_check.py
   ```
2. Phase B (teachers, then runs). See the header of `scripts/09_phase_b.py` for the stages and plans (`students`, `primary`, `pivot`), or use `scripts/run_phaseB_gpu0.sh` and `scripts/run_phaseB_gpu1.sh`. Interrupted runs resume automatically: finished jobs are skipped.
3. Analysis of Phase B:
   ```bash
   python scripts/10_analysis.py
   ```
4. Phase B2a (student alone and vanilla KD on the 27 conditions, seeds 15 to 24, same teachers):
   ```bash
   python scripts/09_phase_b.py --stage run --plan b2a --dataset organmnist3d --smoke   # quick test
   bash scripts/run_b2a_gpu0.sh
   bash scripts/run_b2a_gpu1.sh
   python scripts/13_b2a_analysis.py
   ```
5. Final temperature T at the last epoch (Section 3.7):
   ```bash
   python - <<'E'
   import pandas as pd, glob
   for ds in ["organmnist3d","adrenalmnist3d","fracturemnist3d"]:
       e=pd.concat([pd.read_csv(f,usecols=["epoch","T","method"]) for f in glob.glob(f"outputs/phaseB_{ds}/epochs*.csv")])
       e=e[(e.epoch==e.epoch.max())&e.method.isin(["lin_T","lin_TA","fts","dfs"])]
       print(ds, e.groupby("method").T.agg(["mean","count"]).round(2).to_dict("index"))
   E
   ```

Test data are evaluated once per retained run and are never used to choose a teacher, a student, a checkpoint or a hyperparameter. The Static KD grid and the teachers are chosen on validation data only.

## Rules of the protocol

- The protocol (`fiche_gel_protocole_v1.md`) was frozen before Phase B.
- The addendum for Phase B2a (`addendum_B2a_v1.md`) contains the hypotheses and the classification rules, written before any B2a run. It was not registered externally before the B2a runs started; its content was not changed afterwards. This is declared as a deviation in the paper.
- All deviations are listed in the register of deviations (end of `fiche_gel_protocole_v1.md`) and in the paper.
- Analyses not planned in the protocol are marked as exploratory in the paper.

## License and citation

Code: MIT License (see `LICENSE`). If you use this code, please cite the paper **[reference to be added]**.
