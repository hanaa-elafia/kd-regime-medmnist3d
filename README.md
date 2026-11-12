# Where is knowledge distillation useful for 3D medical image classification?

Code and frozen protocol for an empirical study of knowledge distillation (KD) on three MedMNIST 3D datasets
(OrganMNIST3D, AdrenalMNIST3D, FractureMNIST3D, 28x28x28 voxels, official splits).

We use 3 teachers (3D ResNets), 3 students (small 3D CNNs) and 27 teacher-student-dataset conditions.
We compare several ways to set the temperature T and the loss weight alpha: fixed (vanilla, tuned static),
linear in time, and a fuzzy controller (FTS, FAS, DFS). The study does not propose a new method.
It maps where KD is useful, with a frozen protocol, a protected test set, paired seeds, confidence
intervals, Holm correction and an equivalence margin.

> Status: manuscript in preparation. Paper reference and DOI: **[to be added]**.

## Repository content

| Path | Content |
|---|---|
| `scripts/01_*.py` to `07_acceptance.py` | Phase A: dataset analysis, resolution probe, capacity sweep, teacher and student screening, acceptance checks (validation data only) |
| `scripts/08_pilot_kd.py` | Pilot of the KD schedules (validation data only) |
| `scripts/09_phase_b.py` | Phase B (main experiments) and Phase B2a (confirmation): teachers, runs, status |
| `scripts/09_phase_b_ancienne_phaseB.py` | Version of `09_phase_b.py` used for Phase B, kept for traceability (before the `b2a` plan was added) |
| `scripts/10_analysis.py` | Analysis of Phase B (run once) |
| `scripts/11_kd_wins.py`, `scripts/12_fuzzy_compare.py` | Read-only summaries of Phase B results |
| `scripts/13_b2a_analysis.py` | Analysis of Phase B2a (run once, rules fixed in the addendum) |
| `scripts/14_env_check.py` | Environment check (validation data only) |
| `scripts/run_phaseB_gpu0.sh`, `run_phaseB_gpu1.sh`, `run_b2a_gpu0.sh`, `run_b2a_gpu1.sh` | Launchers for two GPUs |
| `fiche_gel_protocole_v1.md` | Frozen protocol (hypotheses, bounds, statistics, verdict rules, register of deviations) |
| `addendum_B2a_v1.md` | Addendum for Phase B2a (hypotheses HB1 to HB3, classification rules) |
| `environment.yml` | Conda environment used for the experiments |

Not included in this repository: the MedMNIST data (downloaded automatically by the `medmnist` package),
the trained teachers and the run-level result files. They are archived separately:
**[Zenodo link and DOI to be added]**.

## Environment

```bash
conda env create -f environment.yml
conda activate kd-medmnist
```

Experiments ran with Python 3.11.16, PyTorch 2.14.1 (CUDA 13.0 build), NumPy 2.4.6, pandas 3.0.6,
SciPy 1.17.1, scikit-learn 1.9.1 and MedMNIST 3.0.2, on two NVIDIA RTX A4000 GPUs.
GPU determinism was not forced (see the paper), so repeated runs differ slightly.

## How to run

The scripts expect the working directory to contain `scripts/` and `outputs/`.

1. Check the environment (needs the trained teachers in `outputs/phaseB_<dataset>/`):
   ```bash
   python scripts/14_env_check.py
   ```
2. Phase B (teachers, then runs). See the header of `scripts/09_phase_b.py` for the stages and plans
   (`students`, `primary`, `pivot`), or use `scripts/run_phaseB_gpu0.sh` and `scripts/run_phaseB_gpu1.sh`.
   Interrupted runs resume automatically: finished jobs are skipped.
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

Test data are evaluated once per retained run and are never used to choose a teacher, a student,
a checkpoint or a hyperparameter. The Static KD grid and the teachers are chosen on validation data only.

## Rules of the protocol

- The protocol (`fiche_gel_protocole_v1.md`) was frozen before Phase B.
- The addendum for Phase B2a (`addendum_B2a_v1.md`) contains the hypotheses and the classification rules,
  written before any B2a result. It was not dated and signed before the B2a runs started; its content is
  unchanged since it was written (see the register of deviations and the paper).
- Analyses not planned in the protocol are marked as exploratory in the paper.

## License and citation

Code: MIT License (see `LICENSE`). If you use this code, please cite the paper **[reference to be added]**.
