# 多模態圖片弱標籤補強分析統計表格

- 有效貼文數：16,355
- 使用者數：292
- Low alignment：similarity <= 0.2404
- High alignment：similarity >= 0.2881

## 表 1. Text-only / Image-only / Multimodal 比較

| 模型 | 輸入 | AUC | F1 | Accuracy | Precision | Recall | Interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Text-only CLIP | caption text embedding | 0.8529 +/- 0.0385 | 0.7587 +/- 0.0374 | 0.7637 +/- 0.0332 | 0.6964 +/- 0.0395 | 0.8385 +/- 0.0834 | 文字語意基準 |
| Image-only CLIP | image embedding | 0.7927 +/- 0.0311 | 0.7112 +/- 0.0160 | 0.6851 +/- 0.0399 | 0.6077 +/- 0.0514 | 0.8692 +/- 0.0798 | 影像弱標籤訊號 |
| Text+Image concat | text + image concatenation | 尚未完成 | 尚未完成 | 尚未完成 | 尚未完成 | 尚未完成 | 簡單融合 baseline |
| Multimodal ContextVecNet (CLIP/CLIP) | text + image + cross-attention + temporal module | 0.8525 +/- 0.0455 | 0.7316 +/- 0.0329 | 0.7259 +/- 0.0450 | 0.6577 +/- 0.0658 | 0.8385 +/- 0.0996 | 原始 CLIP/CLIP 多模態架構 baseline |
| Text-only BERT-base Chinese | BERT caption embedding | 0.8995 +/- 0.0411 | 0.7801 +/- 0.0329 | 0.7977 +/- 0.0423 | 0.7821 +/- 0.1180 | 0.8077 +/- 0.1304 | 強文字 baseline |
| BERT+Image CLIP | BERT text + CLIP image | 0.9033 +/- 0.0404 | 0.7799 +/- 0.0595 | 0.7909 +/- 0.0760 | 0.7747 +/- 0.1425 | 0.8231 +/- 0.1429 | 主要多模態比較模型 |

## 表 2. CLIP image-text similarity 統計

| 群組 | N | Mean | SD | Median | Q1 | Q3 |
| --- | --- | --- | --- | --- | --- | --- |
| 全部有效貼文 | 16355 | 0.2643 | 0.0347 | 0.2680 | 0.2404 | 0.2881 |
| Positive users | 7263 | 0.2655 | 0.0336 | 0.2713 | 0.2418 | 0.2888 |
| Negative users | 9092 | 0.2634 | 0.0355 | 0.2651 | 0.2394 | 0.2875 |
| TP | 5942 | 0.2683 | 0.0314 | 0.2744 | 0.2487 | 0.2896 |
| FP | 2146 | 0.2646 | 0.0334 | 0.2698 | 0.2418 | 0.2886 |
| FN | 1321 | 0.2531 | 0.0398 | 0.2491 | 0.2245 | 0.2786 |
| TN | 6946 | 0.2630 | 0.0361 | 0.2635 | 0.2386 | 0.2869 |
| High alignment | 4089 | 0.3055 | 0.0192 | 0.3002 | 0.2933 | 0.3108 |
| Medium alignment | 8177 | 0.2665 | 0.0135 | 0.2680 | 0.2551 | 0.2781 |
| Low alignment | 4089 | 0.2187 | 0.0171 | 0.2228 | 0.2097 | 0.2319 |

## 表 3. Alignment group 下的使用者層級模型表現

| Alignment group | Model | N users | AUC | F1 | Accuracy | Precision | Recall | Interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| High | Text-only BERT | 73 | 0.9333 | 0.8736 | 0.8493 | 0.8085 | 0.9500 |  |
| High | BERT+Image CLIP | 73 | 0.9303 | 0.8444 | 0.8082 | 0.7600 | 0.9500 | 圖文語意較一致；本資料中 Text-only BERT 表現較高，顯示文字語意已足夠強，圖片未必進一步提升。 |
| Medium | Text-only BERT | 145 | 0.9197 | 0.7833 | 0.8207 | 0.7344 | 0.8393 |  |
| Medium | BERT+Image CLIP | 145 | 0.9222 | 0.7931 | 0.8345 | 0.7667 | 0.8214 | 圖文有部分關聯；BERT+Image CLIP 的 AUC/F1 略高，代表圖片可能提供有限輔助。 |
| Low | Text-only BERT | 73 | 0.7886 | 0.6333 | 0.6986 | 0.7037 | 0.5758 |  |
| Low | BERT+Image CLIP | 73 | 0.7909 | 0.6567 | 0.6849 | 0.6471 | 0.6667 | 圖文語意較不一致；BERT+Image CLIP 的 recall/F1 略高但 precision 較低，顯示圖片可能同時帶來輔助與噪音。 |

## 表 4. 圖片影響分數 delta_p 統計

| 群組 | N | Mean delta_p | SD | Median | Min | Max | |delta_p| >= 0.10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 全部使用者 | 292 | 0.0044 | 0.0758 | 0.0101 | -0.2927 | 0.3001 | 45 |
| Positive users | 130 | 0.0089 | 0.0765 | 0.0123 | -0.2438 | 0.3001 | 23 |
| Negative users | 162 | 0.0007 | 0.0752 | 0.0084 | -0.2927 | 0.2924 | 22 |
| High alignment users | 73 | 0.0056 | 0.0866 | 0.0144 | -0.2927 | 0.3001 | 11 |
| Medium alignment users | 145 | 0.0075 | 0.0745 | 0.0106 | -0.2821 | 0.2190 | 25 |
| Low alignment users | 73 | -0.0038 | 0.0671 | 0.0033 | -0.2438 | 0.1247 | 9 |
| TP | 107 | 0.0180 | 0.0708 | 0.0161 | -0.2438 | 0.3001 | 16 |
| FP | 38 | 0.0217 | 0.0876 | 0.0294 | -0.1374 | 0.2924 | 7 |
| FN | 23 | -0.0332 | 0.0889 | -0.0236 | -0.2192 | 0.2186 | 7 |
| TN | 124 | -0.0058 | 0.0701 | 0.0005 | -0.2927 | 0.1757 | 15 |

## 表 5. |delta_p| 最大案例 Top 20

| fold | author | label | text_probability | multimodal_probability | delta_p | user_alignment_group | multimodal_error_type |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | painwords.yan | 1 | 0.4776 | 0.7777 | 0.3001 | High | TP |
| 3 | huhuzo1225 | 0 | 0.4518 | 0.1592 | -0.2927 | High | TN |
| 0 | who.loved.by.jesus | 0 | 0.5154 | 0.8077 | 0.2924 | High | FP |
| 1 | hao_emo520 | 0 | 0.6046 | 0.3225 | -0.2821 | Medium | TN |
| 4 | shan_min33 | 1 | 0.9093 | 0.6656 | -0.2438 | Low | TP |
| 3 | mindfullyhk | 1 | 0.4415 | 0.2223 | -0.2192 | High | FN |
| 1 | mu._.studaily | 1 | 0.2034 | 0.4224 | 0.2190 | Medium | TP |
| 3 | will_be._.fine | 1 | 0.1369 | 0.3555 | 0.2186 | Medium | FN |
| 2 | poem_chao | 1 | 0.5977 | 0.7936 | 0.1959 | Medium | TP |
| 3 | chrislin_0208 | 0 | 0.3809 | 0.1952 | -0.1856 | Low | TN |
| 3 | ryouliao | 0 | 0.0376 | 0.2133 | 0.1757 | Medium | TN |
| 2 | psylee01 | 0 | 0.2708 | 0.4366 | 0.1658 | Medium | FP |
| 3 | yi_wen_even | 1 | 0.3602 | 0.1954 | -0.1648 | Medium | FN |
| 0 | yuehchungsu | 0 | 0.3371 | 0.1771 | -0.1599 | Medium | TN |
| 3 | thepetpope | 0 | 0.1921 | 0.3514 | 0.1594 | Medium | TN |
| 3 | powxd01 | 0 | 0.3349 | 0.1764 | -0.1584 | Medium | TN |
| 4 | opheliecc | 1 | 0.7321 | 0.5765 | -0.1555 | High | TP |
| 0 | sunrise1225 | 1 | 0.7509 | 0.6039 | -0.1470 | Low | TP |
| 3 | cancerfighter211 | 1 | 0.7016 | 0.8460 | 0.1444 | Medium | TP |
| 3 | zoezoeliao | 1 | 0.3232 | 0.1835 | -0.1397 | Low | FN |

## 表 6. 圖文不一致案例抽樣統計

| 案例類型 | 案例數 | 建議用途 |
| --- | --- | --- |
| high_alignment_correct | 20 | 圖文一致且模型正確，用於說明圖片與文字一致時的代表案例。 |
| low_alignment_correct | 20 | 圖文不一致但模型正確，用於檢查模型是否主要依賴文字訊號。 |
| low_alignment_error | 20 | 圖文不一致且模型錯誤，用於分析影像噪音或弱標籤限制。 |
| multimodal_false_positive | 20 | 多模態 FP，用於分析圖片是否推高負類風險。 |
| largest_abs_delta_p | 20 | 加入圖片後風險分數變化最大，用於分析圖片影響方向。 |
| multimodal_false_negative | 20 | 多模態 FN，用於分析圖片是否稀釋正類文字訊號。 |
| high_alignment_error | 14 | 圖文一致但模型錯誤，用於檢查即使一致仍誤判的限制。 |

## 表 7. 圖文不一致 taxonomy 人工檢視模板

| 類型 | 說明 | 常見情境 | 可能影響 |
| --- | --- | --- | --- |
| A | Caption 高風險、圖片中性 | 文字有憂鬱語意，但圖片是風景、食物、日常物品 | 圖片可能稀釋文字訊號 |
| B | Caption 中性、圖片低落氛圍 | 文字無明顯風險，但圖片暗色、孤獨、空景 | 圖片可能提高風險分數 |
| C | 圖片為裝飾或無關內容 | 貼圖、廣告、品牌照、與 caption 主題無關 | 影像模態可能成為噪音 |
| D | 圖片含文字截圖 | 圖片本身包含文字訊息，如限動、對話截圖 | CLIP image encoder 未必能完整理解，可考慮 OCR |
| E | 圖文反諷或語氣不一致 | caption 有幽默、反諷或情緒轉折 | 模型容易誤解真實語意 |
| F | 多圖或影片代表影格不足 | 第一張圖片或中間影格不能代表整篇貼文 | 影像特徵不完整 |
| G | 缺圖或低品質圖片 | 黑圖、模糊、過暗、截斷 | 影像訊號可靠度低 |
| H | 平台風格影響 | 濾鏡、自拍、打卡、品牌風格 | 模型可能學到非心理訊號 |

## 可搭配圖表

- `clip_similarity_histogram.png`
- `similarity_by_error_type_boxplot.png`
- `alignment_group_auc_bar.png`
