#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/angle/miniconda3/envs/contextvecnet/bin/python}"
GROUP="${GROUP:-final_preprocessing_v2}"
RESULTS_ROOT="${RESULTS_ROOT:-results/final_preprocessing_v2}"
FOLDS="${FOLDS:-0 1 2 3 4}"
BATCH_SIZE="${BATCH_SIZE:-2}"
CONFIG_FILE="${CONFIG_FILE:-configs/combos/multi_only.yaml}"
OUTPUT_DIR="${RESULTS_ROOT}/supplemental/contextvecnet_uncalibrated_validation_f1/w64"

mkdir -p "${OUTPUT_DIR}"

for fold in ${FOLDS}; do
  name="contextvecnet_w64_fold${fold}"
  "${PYTHON_BIN}" evaluate_calibration_comparison.py \
    --config_file "${CONFIG_FILE}" \
    --name "${name}" \
    --group "${GROUP}" \
    --fold "${fold}" \
    --window_size 64 \
    --batch_size "${BATCH_SIZE}" \
    --output_dir "${OUTPUT_DIR}" \
    --calibrations none \
    --threshold_strategy validation_f1
done

"${PYTHON_BIN}" aggregate_final_experiments.py \
  --results_root "${RESULTS_ROOT}"

echo
echo "Uncalibrated + validation-F1 results:"
"${PYTHON_BIN}" -c \
  "import pandas as pd; p='${RESULTS_ROOT}/all_results_summary.csv'; d=pd.read_csv(p); print(d[(d['method']=='contextvecnet') & (d['window_size']==64) & (d['calibration']=='none') & (d['threshold_strategy']=='validation_f1')].to_string(index=False))"
