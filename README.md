# ContextVecNet Instagram 憂鬱風險實驗交接版

本 repository 是論文 **「繁體中文 Instagram 使用者層級憂鬱風險訊號之多模態時間序列建模」** 的交接版本。此版本基於原始 ContextVecNet 程式碼修改，並整理成本研究使用的 Instagram 使用者層級實驗流程，包含 14 天弱標籤分析、資料前處理、模型訓練、校準比較、模態比較、視窗大小分析、穩健性分析與最終結果表。

## 內容概要

```text
configs/                         模型與實驗設定檔
callbacks/                       訓練 callbacks
clip/                            本研究使用的本地 CLIP 實作
datasets/                        Dataset loader 與 window-level dataset 程式
evaluators/                      評估工具
loggers/                         logging wrapper
models/                          ContextVecNet 與相關模型模組
trainer/                         訓練流程
particular_model_trainers/       原始專案中的 trainer 相關模組
results/                         最終 CSV 結果、預測檔、reliability 圖與 robustness 結果
d_time_series_handoff/           從 D:\時間序列 複製進來的小型程式、notebook 與標註檔
FINAL_EXPERIMENTS.md             最終實驗執行指令與參數說明
DATA.md                          大型資料夾、外部資料放置位置與不上傳原因
```

以下腳本是原作者版本的舊實驗入口，**不是本論文主要實驗流程**：

```bash
experiments/run_experiments.sh
```

## 主要實驗入口

本研究主要實驗驅動腳本是：

```bash
./run_final_experiments.sh
```

常用執行方式如下。完整說明請看 `FINAL_EXPERIMENTS.md`。

```bash
DRY_RUN=1 ./run_final_experiments.sh
ACTION=train ./run_final_experiments.sh
ACTION=evaluate ./run_final_experiments.sh
FOLDS="0" EPOCHS=2 PATIENCE=2 ./run_final_experiments.sh
SECTIONS="methods" METHODS="contextvecnet text_bert" ./run_final_experiments.sh
```

輔助實驗腳本：

```bash
./run_robustness_supplement.sh
./run_v2_uncalibrated_validation_f1.sh
```

## 環境設定

建議使用 Conda 建立環境：

```bash
conda env create -f env.yml
conda activate contextvecnet
```

`env.yml` 比 `requirements.txt` 完整，包含最終實驗會用到的 `scipy`、`matplotlib`、`ftfy`、`regex`、`wandb` 等套件。

如果 shell script 裡預設的 Python 路徑不符合新機器，可以用 `PYTHON_BIN` 指定目前環境的 Python：

```bash
PYTHON_BIN=$(which python) DRY_RUN=1 ./run_final_experiments.sh
```

## 資料說明

完整 Instagram 資料集、圖片資料夾、大型中間特徵檔與 checkpoint **沒有放在 GitHub**。原因是檔案太大，而且可能包含 Instagram 使用者內容或衍生資料，需用私人方式交接，例如外接硬碟、Google Drive、OneDrive 或實驗室伺服器。

外部資料建議放在這些被 `.gitignore` 忽略的位置：

```text
data/
raw_data/
processed_data/
ContextVecNet_Instagram*/
MultiModalDataset/
checkpoints/
```

詳細資料夾用途與建議放置方式請看 `DATA.md`。

## 最終結果

本 repo 已包含最終實驗的主要結果，位置如下：

```text
results/final_preprocessing_v2/
results/robustness_supplement/
results/baselines/
interpretability_outputs/
attention_outputs/
timeline_outputs/
timeline_instability_outputs/
```

常用摘要檔：

```text
results/final_preprocessing_v2/all_results_summary.csv
results/final_preprocessing_v2/method_and_modality_comparison.csv
results/final_preprocessing_v2/calibration_comparison.csv
results/final_preprocessing_v2/window_size_comparison.csv
results/robustness_supplement/label_noise/label_noise_summary.csv
results/robustness_supplement/input_perturbations/bert_clip/w64/input_perturbation_summary.csv
```

這些結果讓接手者即使沒有立即重跑 GPU 實驗，也能檢查論文中的主要表格、預測結果與補充分析。

## 弱標籤與前處理交接資料

從 `D:\時間序列` 複製進來的小型程式、notebook、標註 CSV 與小型 window-level artifacts 放在：

```text
d_time_series_handoff/
```

此資料夾包含弱標籤敏感度分析、資料集過濾、使用者與貼文數統計、人工標註樣本、DER prime 與人工標註比較，以及小型 window-level 資料。大型原始資料與大型中間資料沒有放進 GitHub，而是記錄在 `DATA.md`。

## 建議交接流程

1. Clone 此 repository。
2. 使用 `env.yml` 建立 Conda 環境。
3. 向專案交接者取得外部資料集與 checkpoint。
4. 依照 `DATA.md` 將外部資料放到建議位置。
5. 先執行 dry run，確認路徑與指令展開是否正常：

```bash
PYTHON_BIN=$(which python) DRY_RUN=1 ./run_final_experiments.sh
```

6. 再執行一個小型 smoke test：

```bash
PYTHON_BIN=$(which python) FOLDS="0" EPOCHS=2 PATIENCE=2 ./run_final_experiments.sh
```

7. 若 smoke test 正常，再依照 `FINAL_EXPERIMENTS.md` 執行完整訓練、評估與彙整。

## Attribution

本研究基於原始 ContextVecNet 實作進行修改，並將其應用於繁體中文 Instagram 使用者層級憂鬱風險建模、14 天弱標籤建構、校準分析、模態比較、視窗大小分析與穩健性分析。
