# ContextVecNet Instagram Depression Risk Experiments

This repository is the handoff version of the thesis project **Multimodal Time-Series Modeling of User-Level Depression Risk Signals from Traditional Chinese Instagram Data**. It is based on the original ContextVecNet codebase, but the main experiment flow, dataset preparation, weak-label analysis, calibration evaluation, robustness checks, and result tables were adapted for this Instagram user-level study.

## What Is Included

```text
configs/                         Model and experiment configuration files
callbacks/                       Training callbacks
clip/                            Local CLIP implementation used by the model
datasets/                        Dataset loaders and window-level dataset code
evaluators/                      Evaluation utilities
loggers/                         Logging wrapper
models/                          ContextVecNet and supporting model modules
trainer/                         Training loop implementation
particular_model_trainers/       Original trainer-related modules
results/                         Final CSV tables, predictions, reliability plots, and robustness outputs
d_time_series_handoff/           Small scripts, notebooks, and annotation files copied from D:\時間序列
FINAL_EXPERIMENTS.md             Detailed commands for the final experiments
DATA.md                          Dataset placement, excluded large folders, and handoff notes
```

The old script below is from the original repository and is **not** the main entry point for this thesis project:

```bash
experiments/run_experiments.sh
```

## Main Experiment Entry Point

The primary experiment driver is:

```bash
./run_final_experiments.sh
```

It supports the environment variables documented in `FINAL_EXPERIMENTS.md`, including:

```bash
DRY_RUN=1 ./run_final_experiments.sh
ACTION=train ./run_final_experiments.sh
ACTION=evaluate ./run_final_experiments.sh
FOLDS="0" EPOCHS=2 PATIENCE=2 ./run_final_experiments.sh
SECTIONS="methods" METHODS="contextvecnet text_bert" ./run_final_experiments.sh
```

Auxiliary scripts:

```bash
./run_robustness_supplement.sh
./run_v2_uncalibrated_validation_f1.sh
```

## Environment

The recommended setup is the Conda environment:

```bash
conda env create -f env.yml
conda activate contextvecnet
```

`env.yml` is more complete than `requirements.txt` and includes packages used by the final scripts, such as `scipy`, `matplotlib`, `ftfy`, `regex`, and `wandb`.

If the Python path in the shell scripts does not match the new machine, override it:

```bash
PYTHON_BIN=$(which python) DRY_RUN=1 ./run_final_experiments.sh
```

## Data

Large Instagram datasets, image folders, intermediate feature files, and checkpoints are **not included in GitHub** because of privacy and file-size constraints. See `DATA.md` for the expected folder layout and the external handoff items.

In short, place external data under ignored folders such as:

```text
data/
raw_data/
processed_data/
ContextVecNet_Instagram*/
MultiModalDataset/
checkpoints/
```

These folders are intentionally ignored by `.gitignore`.

## Final Results

Final experiment outputs are stored under:

```text
results/final_preprocessing_v2/
results/robustness_supplement/
results/baselines/
```

Important summary files include:

```text
results/final_preprocessing_v2/all_results_summary.csv
results/final_preprocessing_v2/method_and_modality_comparison.csv
results/final_preprocessing_v2/calibration_comparison.csv
results/final_preprocessing_v2/window_size_comparison.csv
results/robustness_supplement/label_noise/label_noise_summary.csv
results/robustness_supplement/input_perturbations/bert_clip/w64/input_perturbation_summary.csv
```

## Weak-Label And Preprocessing Handoff

Small scripts and notebooks copied from `D:\時間序列` are stored in:

```text
d_time_series_handoff/
```

This folder includes weak-label sensitivity scripts, dataset filtering/counting scripts, annotation samples, and small window-level artifacts. Larger raw or processed datasets are documented in `DATA.md` instead of being committed to GitHub.

## Recommended Handoff Workflow

1. Clone this repository.
2. Create the Conda environment with `env.yml`.
3. Obtain the external dataset/checkpoint package from the project owner.
4. Place external folders according to `DATA.md`.
5. Run a dry-run command first:

```bash
PYTHON_BIN=$(which python) DRY_RUN=1 ./run_final_experiments.sh
```

6. Run a small smoke test before full training:

```bash
PYTHON_BIN=$(which python) FOLDS="0" EPOCHS=2 PATIENCE=2 ./run_final_experiments.sh
```

7. Use `FINAL_EXPERIMENTS.md` for full training, evaluation, and aggregation commands.

## Attribution

This project is based on the original ContextVecNet implementation and adapts it for Traditional Chinese Instagram user-level depression risk modeling, 14-day weak-label construction, calibration analysis, modality comparison, window-size analysis, and robustness evaluation.
