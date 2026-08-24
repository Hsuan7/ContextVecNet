# Data Handoff Notes

This repository intentionally does not include the full Instagram dataset, large image folders, intermediate feature tensors, or checkpoints. Those files should be transferred separately through a private storage channel, such as an external drive, Google Drive, OneDrive, or a lab server.

## Why Large Data Is Not In GitHub

The excluded folders are too large for normal GitHub use, and several contain Instagram user content or derived user-level data. Keeping them outside the public repository reduces privacy risk and prevents clone/push failures caused by large files.

GitHub should contain code, configuration files, documentation, small CSV summaries, figures, and reproducible result tables. Large raw data and model artifacts should be stored separately.

## External Data From `D:\時間序列`

The following folders were found in `D:\時間序列` and should be treated as external handoff data rather than GitHub-tracked files:

| Folder | Approx. size | Suggested placement in this repo | Purpose / note |
|---|---:|---|---|
| `ContextVecNet_Instagram` | not copied | `data/ContextVecNet_Instagram/` | Original Instagram dataset workspace. Excluded from GitHub. |
| `ContextVecNet_Instagram_filtered_new` | not copied | `data/ContextVecNet_Instagram_filtered_new/` | Final filtered Instagram dataset used for user/post metadata checks and model input preparation. Excluded from GitHub. |
| `depress_dataset` | 129 GB | external storage only, or `data/depress_dataset/` if local space allows | Very large raw/derived dataset. Do not commit. |
| `final_model_inputs_vision_all` | 45 GB | `processed_data/final_model_inputs_vision_all/` | Large vision/model input artifacts. Do not commit. |
| `DECEN` | 1.6 GB | `processed_data/DECEN/` | Large intermediate data. Do not commit. |
| `DECEN_TS` | 796 MB | `processed_data/DECEN_TS/` | Time-series / weak-label input workspace. Do not commit. |
| `labeled` | 772 MB | `processed_data/labeled/` | Weak-label outputs and related artifacts. Do not commit as a full folder. |
| `final_preprocessed_data_all` | 537 MB | `processed_data/final_preprocessed_data_all/` | Large preprocessed data. Do not commit. |
| `final_model_inputs_text_time_all` | 354 MB | `processed_data/final_model_inputs_text_time_all/` | Large text/time model inputs. Do not commit. |
| `final_preprocessed_data` | 100 MB | `processed_data/final_preprocessed_data/` | Preprocessed data. Usually better kept outside GitHub. |

The `.gitignore` file already ignores common local data locations:

```text
data/
raw_data/
processed_data/
MultiModalDataset/
ContextVecNet_Instagram*/
checkpoints/
final_model/
*.ckpt
*.pt
*.pth
*.tgz
```

## Small Files Copied Into This Repository

Small scripts, notebooks, annotation CSVs, and small window-level artifacts from `D:\時間序列` were copied into:

```text
d_time_series_handoff/
```

This folder includes files such as:

```text
count_csv_users_posts.py
dataset.py
filter_dataset_no_blank_image.py
label_sensitivity_292_users.py
label_sensitivity_from_input_csv.py
window.py
sentence_level_sample_for_annotation.csv
sentence_level_sample_for_annotation_2.csv
sentence_level_sample_with_label.csv
human_vs_DER_prime_confusion_matrix.png
DER'_human_label/
window_level/
window_level_8/
window_level_plus_prev_context_8/
window_level_plus_prev_context_32/
```

These files are included because they are small enough for GitHub and useful for understanding preprocessing, weak-label construction, annotation review, and dataset statistics.

## Expected Local Setup For Reproduction

After cloning the repository, create the environment and place external data locally:

```bash
conda env create -f env.yml
conda activate contextvecnet
```

Recommended local layout:

```text
ContextVecNet/
  data/
    ContextVecNet_Instagram/
    ContextVecNet_Instagram_filtered_new/
  processed_data/
    DECEN_TS/
    labeled/
    final_model_inputs_text_time_all/
    final_model_inputs_vision_all/
  checkpoints/
    final_preprocessing_v2:<method>_w<window>_fold<fold>/
```

If the scripts expect a different absolute path from the original machine, update the relevant config file or pass the dataset path through the script arguments where supported.

## Results Already Included

The repository includes final result summaries and figures under:

```text
results/final_preprocessing_v2/
results/robustness_supplement/
results/baselines/
```

These allow readers to inspect the reported experimental outcomes without rerunning all GPU experiments.
