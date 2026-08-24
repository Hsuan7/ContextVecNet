# Final Experiments

All results from the corrected text and image preprocessing pipeline are stored
below:

```text
results/final_preprocessing_v1/
  methods/
    contextvecnet/w64/
    text_bert/w64/
    text_clip/w64/
    image_clip/w64/
    concat/w64/
    lstm/w64/
  window_sizes/
    contextvecnet/w16/
    contextvecnet/w32/
    contextvecnet/w64/
    contextvecnet/w128/
  all_results_by_fold.csv
  all_results_summary.csv
  method_and_modality_comparison.csv
  calibration_comparison.csv
  window_size_comparison.csv
  calibration_reliability/
    contextvecnet_w64_calibration_5fold_calibration_curve.png
    contextvecnet_w64_calibration_5fold_reliability_diagram.png
```

Each fold produces:

- Full multimodal ContextVecNet: none, temperature, and Platt calibration
  results.
- Other methods: only the final Platt-calibrated result.
- `*_predictions.csv`: user-level logits, probabilities, and predictions.
- `*_reliability_diagram.png`: per-fold reliability diagram.
- `*_reliability_bins.csv`: bin confidence, observed rate, gap, and count.

The three comparison tables correspond directly to the final experiments:

- `method_and_modality_comparison.csv`: the seven requested comparison rows at
  window size 64. It includes both calibrated and uncalibrated ContextVecNet;
  the other learned methods use their Platt-calibrated result.
- `calibration_comparison.csv`: uncalibrated, temperature scaling, and Platt
  scaling for full multimodal ContextVecNet at window size 64 only. It reports
  ECE-10, Brier score, NLL, AUC, F1, precision, recall, and accuracy.
- `window_size_comparison.csv`: full ContextVecNet at 16, 32, 64, and 128,
  using Platt-calibrated results.

Checkpoints use the group `final_preprocessing_v1`, so they are separate from
older experiments:

```text
checkpoints/final_preprocessing_v1:<method>_w<window>_fold<fold>/
```

## Full Run

From `/home/angle/ContextVecNet`:

```bash
./validate_final_experiments.py
chmod +x run_final_experiments.sh
./run_final_experiments.sh
```

This trains and evaluates:

- Full ContextVecNet
- Text-only Chinese MentalBERT (`zwzzz/Chinese-MentalBERT`)
- Text-only CLIP
- Image-only CLIP
- Text+Image concat
- BiLSTM baseline
- Full ContextVecNet with window sizes 16, 32, 64, and 128

All experiments use five folds. Window size 64 is reused between the method and
window-size comparisons.

## Recommended Staged Run

Preview every command without running it:

```bash
DRY_RUN=1 ./run_final_experiments.sh
```

Train first:

```bash
ACTION=train ./run_final_experiments.sh
```

Evaluate and aggregate after training:

```bash
ACTION=evaluate ./run_final_experiments.sh
```

Run one fold first as a smoke test:

```bash
FOLDS="0" EPOCHS=2 PATIENCE=2 ./run_final_experiments.sh
```

Run selected methods:

```bash
SECTIONS="methods" METHODS="contextvecnet text_bert" ./run_final_experiments.sh
```

Adjust resource settings:

```bash
BATCH_SIZE=1 EPOCHS=50 PATIENCE=50 ./run_final_experiments.sh
```

## Separate Runs

Run only the method and modality comparison:

```bash
SECTIONS="methods" ./run_final_experiments.sh
```

Run one method at a time:

```bash
SECTIONS="methods" METHODS="contextvecnet" ./run_final_experiments.sh
SECTIONS="methods" METHODS="text_bert" ./run_final_experiments.sh
SECTIONS="methods" METHODS="text_clip" ./run_final_experiments.sh
SECTIONS="methods" METHODS="image_clip" ./run_final_experiments.sh
SECTIONS="methods" METHODS="concat" ./run_final_experiments.sh
SECTIONS="methods" METHODS="lstm" ./run_final_experiments.sh
```

The ContextVecNet method command automatically evaluates none, temperature,
and Platt calibration. Other methods output only their Platt result.

Run only the window-size comparison:

```bash
SECTIONS="window_sizes" ./run_final_experiments.sh
```

Run one window size at a time:

```bash
SECTIONS="window_sizes" WINDOW_SIZES="16" ./run_final_experiments.sh
SECTIONS="window_sizes" WINDOW_SIZES="32" ./run_final_experiments.sh
SECTIONS="window_sizes" WINDOW_SIZES="64" ./run_final_experiments.sh
SECTIONS="window_sizes" WINDOW_SIZES="128" ./run_final_experiments.sh
```

Run one fold at a time:

```bash
FOLDS="0" ./run_final_experiments.sh
FOLDS="1" ./run_final_experiments.sh
FOLDS="2" ./run_final_experiments.sh
FOLDS="3" ./run_final_experiments.sh
FOLDS="4" ./run_final_experiments.sh
```

Training and evaluation can also be separated for any command:

```bash
SECTIONS="methods" METHODS="text_bert" ACTION=train ./run_final_experiments.sh
SECTIONS="methods" METHODS="text_bert" ACTION=evaluate ./run_final_experiments.sh
```

## Calibration Protocol

For each full multimodal ContextVecNet checkpoint:

1. Collect validation logits.
2. Fit temperature scaling on validation NLL.
3. Fit Platt scaling on validation logits.
4. Compare all three calibration methods using the same fixed threshold 0.5.
5. Apply the frozen calibration parameters to the test fold.

The test fold is never used to fit calibration or select thresholds.
Output rows explicitly record `calibration_fit_split`,
`threshold_selection_split`, and `evaluation_split` for auditing.

The calibration comparison is reported only for full multimodal ContextVecNet.
The window-size comparison also contains only full multimodal ContextVecNet.
For baseline and window-size final results, the threshold is selected on the
validation fold by F1 and then frozen for the test fold.
