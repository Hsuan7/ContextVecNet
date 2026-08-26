# 模型權重與 Checkpoint 交接說明

本 repository 沒有把完整訓練模型權重放進 GitHub。原因是多數 `.ckpt`、`.pt`、`.safetensors` 單檔大小約 400MB 到 1GB，超過一般 GitHub 使用範圍，也會讓 clone/push 非常困難。

如果接手者只需要查看論文結果，可以直接使用 GitHub 內的：

```text
results/
interpretability_outputs/
attention_outputs/
timeline_outputs/
timeline_instability_outputs/
llm_handoff/
```

如果接手者要重新執行 `ACTION=evaluate` 或重現模型預測，請另外取得以下外部模型資料。

## ContextVecNet / BERT+CLIP Checkpoint

原始位置：

```text
/home/angle/ContextVecNet/checkpoints/
```

建議在新環境放置為：

```text
ContextVecNet/checkpoints/
```

重要 checkpoint 群組包含：

- `checkpoints/final_preprocessing_v2:bert_clip_w64_fold*/`
- `checkpoints/final_preprocessing_v2:bert_clip_w16_fold*/`
- `checkpoints/final_preprocessing_v2:bert_clip_w32_fold*/`
- `checkpoints/final_preprocessing_v2:bert_clip_w128_fold*/`
- `checkpoints/final_preprocessing_v2:contextvecnet_w64_fold*/`
- `checkpoints/final_preprocessing_v2:text_bert_w64_fold*/`
- `checkpoints/final_preprocessing_v2:text_clip_w64_fold*/`
- `checkpoints/final_preprocessing_v2:image_clip_w64_fold*/`
- `checkpoints/final_preprocessing_v2:concat_w64_fold*/`
- `checkpoints/final_preprocessing_v2:lstm_w64_fold*/`

## `final_model/` 舊版整理資料

原始位置：

```text
/home/angle/ContextVecNet/final_model/
```

此資料夾約 18GB，包含較早期整理的模型、baseline table、各方法 checkpoint 與部分結果。若接手者需要完整歷史實驗脈絡，請用外部硬碟或私人雲端交接。

## `D:\llm` 模型

以下 LLM/DER 模型需另外交接，不能放 GitHub：

```text
D:\llm\exp_DER_param_version\best_DER.pt
D:\llm\exp_DER_balanced\best_DER.pt
D:\llm\exp_best_models_bmes\step50\best_model\model.safetensors
```

GitHub 內只保留 `llm_handoff/exp_best_models_bmes_step50_metadata/` 的 config/tokenizer metadata，不包含大型 model weight。

## 驗證方式

放好外部 checkpoint 後，可先執行 dry run：

```bash
PYTHON_BIN=$(which python) DRY_RUN=1 SECTIONS="methods" METHODS="bert_clip" FOLDS="0" ./run_final_experiments.sh
```

若只要測試評估流程，使用：

```bash
PYTHON_BIN=$(which python) SECTIONS="methods" METHODS="bert_clip" FOLDS="0" ACTION=evaluate ./run_final_experiments.sh
```
