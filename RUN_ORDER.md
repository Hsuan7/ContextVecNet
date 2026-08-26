# 實驗執行順序

本文件依照「資料準備 → 主要模型實驗 → 補充實驗 → 解釋性分析 → LLM 實驗」整理執行方法。實際執行前請先確認 `DATA.md` 與 `MODEL_ARTIFACTS.md` 中的外部資料已放到正確位置。

## 0. 環境設定

```bash
conda env create -f env.yml
conda activate contextvecnet
```

若 shell script 預設 Python 路徑不符合新機器，執行時加上：

```bash
PYTHON_BIN=$(which python)
```

先做 dry run：

```bash
PYTHON_BIN=$(which python) DRY_RUN=1 SECTIONS="methods" METHODS="bert_clip" FOLDS="0" ./run_final_experiments.sh
```

## 1. 資料前處理與弱標籤檢查

相關程式位於：

```text
d_time_series_handoff/
datasets/
```

| 順序 | 程式 | 功能 |
|---:|---|---|
| 1 | `d_time_series_handoff/filter_dataset_no_blank_image.py` | 過濾空白或無效圖片資料。 |
| 2 | `d_time_series_handoff/count_csv_users_posts.py` | 統計使用者數與貼文數。 |
| 3 | `d_time_series_handoff/label_sensitivity_from_input_csv.py` | 從 combined CSV 執行 14 天 sliding-window 弱標籤敏感度分析。 |
| 4 | `d_time_series_handoff/label_sensitivity_292_users.py` | 針對 292 users 版本進行弱標籤敏感度檢查。 |
| 5 | `d_time_series_handoff/window.py` | 早期 window-level 資料產生或檢查程式。 |

大型原始與前處理資料不在 GitHub，請看 `DATA.md`。

## 2. 最終主要模型實驗

主要腳本：

```bash
./run_final_experiments.sh
```

### 2.1 方法與模態比較

一次跑所有方法：

```bash
PYTHON_BIN=$(which python) SECTIONS="methods" ./run_final_experiments.sh
```

逐一跑各方法：

```bash
PYTHON_BIN=$(which python) SECTIONS="methods" METHODS="contextvecnet" ./run_final_experiments.sh
PYTHON_BIN=$(which python) SECTIONS="methods" METHODS="text_bert" ./run_final_experiments.sh
PYTHON_BIN=$(which python) SECTIONS="methods" METHODS="bert_clip" ./run_final_experiments.sh
PYTHON_BIN=$(which python) SECTIONS="methods" METHODS="text_clip" ./run_final_experiments.sh
PYTHON_BIN=$(which python) SECTIONS="methods" METHODS="image_clip" ./run_final_experiments.sh
PYTHON_BIN=$(which python) SECTIONS="methods" METHODS="concat" ./run_final_experiments.sh
PYTHON_BIN=$(which python) SECTIONS="methods" METHODS="lstm" ./run_final_experiments.sh
```

輸出位置：

```text
results/final_preprocessing_v2/methods/
results/final_preprocessing_v2/method_and_modality_comparison.csv
results/final_preprocessing_v2/all_results_summary.csv
```

模型 checkpoint 外部位置：

```text
checkpoints/final_preprocessing_v2:<method>_w64_fold<fold>/
```

### 2.2 只訓練或只評估

只訓練：

```bash
PYTHON_BIN=$(which python) SECTIONS="methods" METHODS="bert_clip" ACTION=train ./run_final_experiments.sh
```

只評估：

```bash
PYTHON_BIN=$(which python) SECTIONS="methods" METHODS="bert_clip" ACTION=evaluate ./run_final_experiments.sh
```

只評估需要外部 checkpoint，詳見 `MODEL_ARTIFACTS.md`。

### 2.3 Window Size 比較

一次跑所有 window size：

```bash
PYTHON_BIN=$(which python) SECTIONS="window_sizes" ./run_final_experiments.sh
```

逐一跑：

```bash
PYTHON_BIN=$(which python) SECTIONS="window_sizes" WINDOW_SIZES="16" ./run_final_experiments.sh
PYTHON_BIN=$(which python) SECTIONS="window_sizes" WINDOW_SIZES="32" ./run_final_experiments.sh
PYTHON_BIN=$(which python) SECTIONS="window_sizes" WINDOW_SIZES="64" ./run_final_experiments.sh
PYTHON_BIN=$(which python) SECTIONS="window_sizes" WINDOW_SIZES="128" ./run_final_experiments.sh
```

預設 window-size 比較方法：

```text
WINDOW_SIZE_METHOD=bert_clip
```

輸出位置：

```text
results/final_preprocessing_v2/window_sizes/
results/final_preprocessing_v2/window_size_comparison.csv
```

## 3. 校準比較

校準比較由 `run_final_experiments.sh` 中的 `evaluate_calibration_comparison.py` 執行。`contextvecnet` 在 methods section 會輸出 `none`、`temperature`、`platt`；其他方法預設使用 `calibrations=none` 與 `threshold_strategy=validation_f1`。

輸出位置：

```text
results/final_preprocessing_v2/calibration_comparison.csv
results/final_preprocessing_v2/calibration_reliability/
```

## 4. Robustness 補充實驗

```bash
PYTHON_BIN=$(which python) ./run_robustness_supplement.sh
```

| 實驗 | 程式 | 輸出 |
|---|---|---|
| label-noise analysis | `robustness_label_noise_analysis.py` | `results/robustness_supplement/label_noise/` |
| input perturbation | `evaluate_input_perturbations.py` | `results/robustness_supplement/input_perturbations/bert_clip/w64/` |
| perturbation aggregation | `aggregate_input_perturbations.py` | `input_perturbation_summary.csv` |

預設 perturbations：

```text
baseline time_shuffle history_mismatch no_history image_mismatch image_zero
```

## 5. 補充：Uncalibrated + Validation-F1

```bash
PYTHON_BIN=$(which python) ./run_v2_uncalibrated_validation_f1.sh
```

輸出位置：

```text
results/final_preprocessing_v2/supplemental/contextvecnet_uncalibrated_validation_f1/w64/
```

## 6. 解釋性與可視化分析

主要結果已在 GitHub：

```text
interpretability_outputs/
attention_outputs/
timeline_outputs/
timeline_instability_outputs/
```

| 順序 | 程式 | 功能 | 主要輸出 |
|---:|---|---|---|
| 1 | `run_bert_clip_xai.py` | 彙整 BERT+CLIP XAI、錯誤分析與代表案例 | `interpretability_outputs/bert_clip_w64_main_xai/` |
| 2 | `extract_attention.py` | 抽取 attention heatmap 與 timeline attention | `attention_outputs/` 或 `interpretability_outputs/.../attention_cases/` |
| 3 | `make_probability_shap.py` | 產生 probability surrogate / SHAP 相關圖表 | `interpretability_outputs/.../probability_shap/` |
| 4 | `make_refined_probability_shap.py` | 產生 refined probability SHAP 圖表 | `interpretability_outputs/.../probability_shap/` |
| 5 | `make_occlusion_attention_plots.py` | 產生 occlusion + attention timeline 圖 | `interpretability_outputs/.../occlusion_attention_cases/` |
| 6 | `make_timeline_visualization.py` | 產生指定使用者 timeline 視覺化 | `timeline_outputs/` |
| 7 | `timeline_instability_analysis.py` | 產生 timeline instability 指標與軌跡 | `timeline_instability_outputs/` |
| 8 | `compare_timeline_trajectories.py` | 比較特定使用者軌跡 | `timeline_instability_outputs/` |

解釋性分析通常需要已訓練 checkpoint 與預測結果。

## 7. Image-Text Alignment 與人工標註分析

| 程式 / 資料夾 | 功能 |
|---|---|
| `generate_alignment_paper_tables.py` | 產生論文用 image-text alignment 表格。 |
| `generate_minimal_case_review_samples.py` | 產生人工檢視用最小案例樣本。 |
| `make_annotation_consensus.py` | 整合人工標註者結果與 consensus。 |
| `annotation_case_review_server.py` | 本地案例檢視 server。 |
| `results/final_preprocessing_v2/image_text_alignment_analysis/` | alignment、case review、標註一致性與表格結果。 |

## 8. LLM / DER 補充實驗

GitHub 內的小型交接資料：

```text
llm_handoff/
```

大型模型與大型 CSV 請外部交接，詳見 `DATA.md` 與 `MODEL_ARTIFACTS.md`。

若 notebook 需要 OpenAI API，請在環境變數設定，不要寫入 notebook：

```bash
export OPENAI_API_KEY="your_key_here"
```

## 9. 結果彙整

```bash
PYTHON_BIN=$(which python) aggregate_final_experiments.py --results_root results/final_preprocessing_v2
```

主要輸出：

```text
results/final_preprocessing_v2/all_results_by_fold.csv
results/final_preprocessing_v2/all_results_summary.csv
results/final_preprocessing_v2/method_and_modality_comparison.csv
results/final_preprocessing_v2/calibration_comparison.csv
results/final_preprocessing_v2/window_size_comparison.csv
```
