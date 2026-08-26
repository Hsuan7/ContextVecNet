# 資料交接說明

本 repository 不包含完整 Instagram 資料集、大型圖片資料夾、大型中間特徵檔或模型 checkpoint。這些檔案請用私人儲存空間另外交接，例如外接硬碟、Google Drive、OneDrive 或實驗室伺服器。

## 為什麼大型資料不放 GitHub

大型資料不放 GitHub 的原因如下：

1. 檔案大小過大，不適合一般 GitHub clone / push。
2. 部分資料可能包含 Instagram 使用者內容、圖片、貼文或衍生使用者層級資料。
3. 將大型原始資料與模型 artifact 放進 Git 會讓 repository 難以維護，也容易觸發 GitHub 檔案大小限制。

GitHub 主要用來保存：程式碼、設定檔、文件、小型 CSV 摘要、圖表、論文用結果表與可重現流程說明。大型原始資料、前處理資料與 checkpoint 則應另外保存。

## 來自 `D:\時間序列` 的外部大型資料

以下資料夾存在於原本的 `D:\時間序列`，但沒有完整放進 GitHub。接手者若要重跑完整流程，需另外取得這些資料。

| 資料夾 | 約略大小 | 建議放置位置 | 用途 / 備註 |
|---|---:|---|---|
| `ContextVecNet_Instagram` | 未複製 | `data/ContextVecNet_Instagram/` | 原始 Instagram dataset workspace。因資料內容與大小限制，不放 GitHub。 |
| `ContextVecNet_Instagram_filtered_new` | 未複製 | `data/ContextVecNet_Instagram_filtered_new/` | 最終 filtered Instagram dataset，用於 metadata 統計與模型輸入準備。因資料內容與大小限制，不放 GitHub。 |
| `depress_dataset` | 約 129 GB | 外部儲存空間，或本機 `data/depress_dataset/` | 非常大型的原始或衍生資料。不要 commit。 |
| `final_model_inputs_vision_all` | 約 45 GB | `processed_data/final_model_inputs_vision_all/` | 大型 vision/model input artifacts。不要 commit。 |
| `DECEN` | 約 1.6 GB | `processed_data/DECEN/` | 大型中間資料。不要 commit。 |
| `DECEN_TS` | 約 796 MB | `processed_data/DECEN_TS/` | 時間序列與弱標籤輸入 workspace。不要 commit。 |
| `labeled` | 約 772 MB | `processed_data/labeled/` | 弱標籤輸出與相關 artifacts。完整資料夾不要 commit。 |
| `final_preprocessed_data_all` | 約 537 MB | `processed_data/final_preprocessed_data_all/` | 大型前處理資料。不要 commit。 |
| `final_model_inputs_text_time_all` | 約 354 MB | `processed_data/final_model_inputs_text_time_all/` | 大型文字與時間模型輸入。不要 commit。 |
| `final_preprocessed_data` | 約 100 MB | `processed_data/final_preprocessed_data/` | 前處理資料。通常不建議放 GitHub。 |

`.gitignore` 已忽略常見的大型資料與模型 artifact 位置：

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

## 已放入 GitHub 的小型交接資料

小型程式、notebook、標註 CSV 與小型 window-level artifacts 已複製到：

```text
d_time_series_handoff/
```

此資料夾包含：

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

這些檔案較小，且有助於理解資料前處理、弱標籤建構、人工標註檢查與資料統計，因此保留在 GitHub。

## 建議本機資料夾結構

clone repository 後，建議將外部資料放成以下結構：

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

若程式或設定檔仍指向原本機器上的絕對路徑，請在新環境中修改對應 config，或在支援參數的 script 中傳入新的 dataset path。

## 已包含的結果資料

此 repository 已包含主要結果摘要與圖表：

```text
results/final_preprocessing_v2/
results/robustness_supplement/
results/baselines/
interpretability_outputs/
attention_outputs/
timeline_outputs/
timeline_instability_outputs/
llm_handoff/
```

這些檔案可讓接手者檢查論文中的主要實驗結果，而不需要立刻重跑所有 GPU 實驗。
