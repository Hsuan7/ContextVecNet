# 專案交接文件

本文件是交接給下一位研究者或工程接手者的總覽。建議先閱讀本文件，再依序閱讀 `README.md`、`RUN_ORDER.md`、`DATA.md`、`MODEL_ARTIFACTS.md`、`PROGRAM_INDEX.md`。

## 一、GitHub Repository

```text
https://github.com/Hsuan7/ContextVecNet
```

本 repo 是論文「繁體中文 Instagram 使用者層級憂鬱風險訊號之多模態時間序列建模」的交接版，包含主要程式碼、小型結果、圖表、解釋性輸出與交接說明。

## 二、建議閱讀順序

1. `README.md`：整體研究、環境與主要入口。
2. `RUN_ORDER.md`：完整執行順序與各實驗命令。
3. `DATA.md`：資料位置、外部資料需求與大型資料不上傳原因。
4. `MODEL_ARTIFACTS.md`：訓練模型權重與 checkpoint 的外部交接清單。
5. `PROGRAM_INDEX.md`：各程式與資料夾用途。
6. `FINAL_EXPERIMENTS.md`：最終實驗腳本的詳細參數與範例。

## 三、Repo 內主要內容

| 項目 | 位置 | 說明 | GitHub 內是否包含 |
|---|---|---|---|
| 主要研究程式碼 | `main_maple.py`, `evaluate_calibration_comparison.py`, `datasets/`, `models/`, `trainer/` | 訓練與評估 ContextVecNet / BERT+CLIP / baseline 模型 | 是 |
| 最終實驗結果 | `results/final_preprocessing_v2/` | 方法比較、校準比較、window-size 比較、預測結果與 reliability 圖 | 是 |
| 穩健性分析 | `results/robustness_supplement/` | label-noise 與 input perturbation 結果 | 是 |
| 解釋性分析 | `interpretability_outputs/` | SHAP、錯誤分析、decision margin、代表案例圖 | 是 |
| Attention 個案 | `attention_outputs/` | attention heatmap 與 timeline attention 圖 | 是 |
| Timeline 視覺化 | `timeline_outputs/`, `timeline_instability_outputs/` | 使用者風險軌跡與 instability 分析 | 是 |
| D 槽小型交接資料 | `d_time_series_handoff/` | 弱標籤、前處理、標註檢查、小型 notebook | 是 |
| LLM 小型交接資料 | `llm_handoff/` | LLM notebook、小型 synthetic 結果、model metadata | 是 |
| 完整 Instagram dataset | `data/` 或外部硬碟 | 大型原始/前處理資料 | 否，需外部交接 |
| 訓練模型 checkpoint | `checkpoints/`, `final_model/`, `D:\llm` | 大型模型權重 | 否，需外部交接 |

## 四、接手者最小復現需求

| 目標 | 需要內容 |
|---|---|
| 檢查論文結果與圖表 | GitHub repo |
| 重新執行評估 | GitHub repo + `checkpoints/` |
| 完整重新訓練 | GitHub repo + 完整資料集 + `processed_data/` + GPU 環境 |
| 重跑 LLM/DER 實驗 | GitHub repo + `D:\llm` 大型 CSV + LLM/DER 模型權重 + `OPENAI_API_KEY` |

## 五、建議執行環境

```text
Windows + WSL2 Ubuntu + Conda + CUDA GPU
```

可以用 VSCode 編輯，但建議用 WSL terminal 執行 `.sh` 腳本，不建議直接用 Windows PowerShell 跑。

## 六、是否需要把程式檔名改成 `1_`, `2_`

不建議。現有腳本與 Python import 已使用固定檔名，改名容易造成找不到檔案或 import error。建議保留原檔名，改用文件管理順序：

```text
RUN_ORDER.md       說明實驗執行順序
PROGRAM_INDEX.md   說明每個程式用途
DATA.md            說明資料位置
MODEL_ARTIFACTS.md 說明模型權重位置
```

## 七、交接時另外提供的外部資料

請用外接硬碟、私人雲端或實驗室伺服器提供：

```text
完整 Instagram dataset
processed_data/
checkpoints/
final_model/
D:\llm 大型模型與大型 CSV
```

詳細清單見 `DATA.md` 與 `MODEL_ARTIFACTS.md`。
