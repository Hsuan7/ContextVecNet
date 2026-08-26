# 程式與資料夾功能索引

本文件列出交接版 repository 中主要程式與資料夾用途。建議不要更改程式檔名；若需要執行順序，請看 `RUN_ORDER.md`。

## 一、主要執行腳本

| 程式 | 功能 | 常見輸出 / 備註 |
|---|---|---|
| `run_final_experiments.sh` | 最終模型訓練、評估與彙整的主腳本。支援 `ACTION`、`SECTIONS`、`METHODS`、`FOLDS`、`WINDOW_SIZES`。 | `results/final_preprocessing_v2/` |
| `run_robustness_supplement.sh` | 執行 label-noise 與 input perturbation 補充實驗。 | `results/robustness_supplement/` |
| `run_v2_uncalibrated_validation_f1.sh` | 補跑 uncalibrated + validation-F1 threshold 的 ContextVecNet 結果。 | `results/final_preprocessing_v2/supplemental/` |
| `experiments/run_experiments.sh` | 原作者舊實驗腳本。 | 非本研究主要入口。 |

## 二、訓練與評估核心程式

| 程式 | 功能 |
|---|---|
| `main_maple.py` | 模型訓練入口，讀取 config、建立 dataset/model/trainer 並訓練。 |
| `evaluate_calibration_comparison.py` | 載入 checkpoint，執行 none / temperature / Platt calibration 評估，輸出 metrics、predictions、reliability 圖。 |
| `evaluate_maple.py` | 較早期或一般評估入口。主要最終流程已改用 `evaluate_calibration_comparison.py`。 |
| `validate_final_experiments.py` | 檢查 final experiment 設定、資料與流程是否可用。 |
| `aggregate_final_experiments.py` | 彙整 final experiment 各 fold 結果，產生 summary 與 comparison CSV。 |
| `utils.py` | 共用工具，例如讀取 args、載入 checkpoint。 |
| `nomenclature.py` | 方法名稱、顯示名稱或表格命名相關設定。 |

## 三、模型與訓練模組

| 路徑 | 功能 |
|---|---|
| `maple_model.py` | ContextVecNet / MaPLe 相關模型組裝。 |
| `models/multimodal_transformer.py` | 多模態 transformer 架構。 |
| `models/time2vec.py` | Time2Vec 時間嵌入模組。 |
| `models/layers/attention.py` | attention 與 cross-attention layer。 |
| `trainer/trainer.py` | 主要訓練流程。 |
| `particular_model_trainers/trainer.py` | 原始專案 trainer 實作。 |
| `particular_model_trainers/trainer_temp.py` | trainer 變體或暫存版本。 |
| `particular_model_trainers/acumen_trainer.py` | 原始專案保留的 trainer 相關程式。 |
| `callbacks/` | early stopping、checkpoint、lambda callback 等訓練 callback。 |
| `loggers/wandb_logger.py` | Weights & Biases logging wrapper。 |

## 四、資料讀取與前處理模組

| 路徑 | 功能 |
|---|---|
| `datasets/twitter_learn.py` | 主要 `CombinedTwitterDataset` dataset loader。雖然名稱保留 Twitter，但本研究流程沿用此 loader 結構讀取 Instagram 資料。 |
| `datasets/window_level_dataset.py` | window-level dataset。 |
| `datasets/time_dataset_learn.py` | 時間序列 dataset 相關程式。 |
| `datasets/preprocessing.py` | 前處理工具。 |
| `d_time_series_handoff/filter_dataset_no_blank_image.py` | 過濾空白圖片或無效圖片資料。 |
| `d_time_series_handoff/count_csv_users_posts.py` | 統計使用者與貼文數。 |
| `d_time_series_handoff/label_sensitivity_from_input_csv.py` | 14 天 sliding-window 弱標籤敏感度分析。 |
| `d_time_series_handoff/label_sensitivity_292_users.py` | 292 users 版本弱標籤敏感度分析。 |
| `d_time_series_handoff/window.py` | 早期 window-level 資料產生或檢查。 |
| `d_time_series_handoff/dataset.py` | D 槽資料處理流程中的 dataset 程式。 |

## 五、Robustness 補充實驗

| 程式 | 功能 | 輸出 |
|---|---|---|
| `robustness_label_noise_analysis.py` | 對 saved predictions 做 label-noise / random-label robustness 分析。 | `results/robustness_supplement/label_noise/` |
| `evaluate_input_perturbations.py` | 執行 baseline、time shuffle、history mismatch、no history、image mismatch、image zero 等 perturbation 評估。 | `results/robustness_supplement/input_perturbations/` |
| `aggregate_input_perturbations.py` | 彙整 input perturbation 各 fold 結果。 | `input_perturbation_summary.csv` |

## 六、解釋性與可視化程式

| 程式 | 功能 | 輸出 |
|---|---|---|
| `run_bert_clip_xai.py` | BERT+CLIP 主 XAI pipeline，彙整錯誤分析、代表案例、SHAP 相關輸出。 | `interpretability_outputs/bert_clip_w64_main_xai/` |
| `extract_attention.py` | 抽取 attention heatmap 與 timeline attention。 | `attention_outputs/` 或 `interpretability_outputs/.../attention_cases/` |
| `make_probability_shap.py` | 產生 probability surrogate 與 SHAP 圖表。 | `interpretability_outputs/.../probability_shap/` |
| `make_refined_probability_shap.py` | 產生 refined probability SHAP 圖表。 | `interpretability_outputs/.../probability_shap/` |
| `make_occlusion_attention_plots.py` | 產生 occlusion contribution 與 attention timeline 圖。 | `interpretability_outputs/.../occlusion_attention_cases/` |
| `make_timeline_visualization.py` | 產生個別使用者 timeline risk/attention 視覺化。 | `timeline_outputs/` |
| `timeline_instability_analysis.py` | 分析不同時間視窗或 timeline 下的風險不穩定性。 | `timeline_instability_outputs/` |
| `compare_timeline_trajectories.py` | 比較特定使用者 risk trajectory。 | `timeline_instability_outputs/` |
| `select_full_timeline_cases.py` | 選取適合完整 timeline 分析的案例。 | case selection CSV |
| `make_reliability_dashboard.py` | 產生 reliability dashboard 相關輸出。 | reliability dashboard / reliability CSV |

## 七、Image-Text Alignment 與人工標註

| 程式 / 資料夾 | 功能 |
|---|---|
| `generate_alignment_paper_tables.py` | 將 image-text alignment analysis 結果整理成論文表格。 |
| `generate_minimal_case_review_samples.py` | 產生人工案例檢視用 sample。 |
| `make_annotation_consensus.py` | 彙整不同標註者結果並產生 consensus。 |
| `annotation_case_review_server.py` | 啟動本地案例檢視 server。 |
| `results/final_preprocessing_v2/image_text_alignment_analysis/` | alignment 指標、case review、標註一致性與論文用表格。 |
| `d_time_series_handoff/DER'_human_label/` | DER prime 與人工標註比較資料。 |

## 八、CLIP 與外部模型工具

| 路徑 | 功能 |
|---|---|
| `clip/clip.py` | CLIP 載入與 preprocessing。 |
| `clip/model.py` | CLIP model 定義。 |
| `clip/simple_tokenizer.py` | CLIP tokenizer。 |
| `clip/bpe_simple_vocab_16e6.txt.gz` | CLIP tokenizer vocabulary。 |

## 九、LLM 交接資料

| 路徑 | 功能 |
|---|---|
| `llm_handoff/few_shot.ipynb` | LLM few-shot 相關 notebook，已移除 API key。 |
| `llm_handoff/llm_3stage.ipynb` | LLM 三階段流程 notebook。 |
| `llm_handoff/synthetic_3stage/` | synthetic 3-stage 小型輸出。 |
| `llm_handoff/merged_stage3_full.csv` | 小型 stage3 合併結果。 |
| `llm_handoff/original_posts_awu.csv` | 小型原始貼文樣本。 |
| `llm_handoff/exp_best_models_bmes_step50_metadata/` | HuggingFace model config/tokenizer metadata，不含大型 weight。 |

大型 LLM 權重與大型 CSV 請看 `DATA.md` 與 `MODEL_ARTIFACTS.md`。

## 十、結果資料夾

| 資料夾 | 功能 |
|---|---|
| `results/final_preprocessing_v2/` | 最終主要實驗結果。 |
| `results/robustness_supplement/` | robustness 補充實驗結果。 |
| `results/baselines/` | baseline 結果。 |
| `interpretability_outputs/` | XAI、SHAP、錯誤分析與代表案例圖。 |
| `attention_outputs/` | attention heatmap 與 timeline attention 圖。 |
| `timeline_outputs/` | 個別 timeline 視覺化。 |
| `timeline_instability_outputs/` | timeline instability 圖與 CSV。 |

## 十一、不在 GitHub 的內容

| 內容 | 原因 | 交接方式 |
|---|---|---|
| 完整 Instagram dataset | 大型且可能含使用者資料 | 外接硬碟 / 私人雲端 |
| `checkpoints/` | `.ckpt` 單檔可達 400MB-1GB | 外接硬碟 / 私人雲端 |
| `final_model/` | 約 18GB | 外接硬碟 / 私人雲端 |
| `D:\llm` 大型模型與大型 CSV | 單檔可達 400MB 以上 | 外接硬碟 / 私人雲端 |
