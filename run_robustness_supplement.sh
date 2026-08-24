#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/angle/miniconda3/envs/contextvecnet/bin/python}"
RESULTS_ROOT="${RESULTS_ROOT:-results/final_preprocessing_v2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/robustness_supplement}"
METHODS="${METHODS:-text_bert bert_clip}"
INPUT_METHOD="${INPUT_METHOD:-bert_clip}"
WINDOW_SIZE="${WINDOW_SIZE:-64}"
FOLDS="${FOLDS:-0 1 2 3 4}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GROUP="${GROUP:-final_preprocessing_v2}"
PERTURBATIONS="${PERTURBATIONS:-baseline time_shuffle history_mismatch no_history image_mismatch image_zero}"

config_for_method() {
  case "$1" in
    text_bert) echo "configs/combos/text_only_bert.yaml" ;;
    bert_clip) echo "configs/combos/bert_clip_contextvecnet.yaml" ;;
    *) echo "No robustness config registered for method: $1" >&2; return 1 ;;
  esac
}

run_command() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if [[ "${DRY_RUN:-0}" != "1" ]]; then
    "$@"
  fi
}

if [[ "${RUN_LABEL_NOISE:-1}" == "1" ]]; then
  run_command "${PYTHON_BIN}" robustness_label_noise_analysis.py \
    --results_root "${RESULTS_ROOT}" \
    --window_size "${WINDOW_SIZE}" \
    --methods ${METHODS} \
    --output_dir "${OUTPUT_ROOT}/label_noise"
fi

if [[ "${RUN_INPUT_PERTURBATIONS:-1}" == "1" ]]; then
  config_file="$(config_for_method "${INPUT_METHOD}")"
  output_dir="${OUTPUT_ROOT}/input_perturbations/${INPUT_METHOD}/w${WINDOW_SIZE}"
  for fold in ${FOLDS}; do
    run_command "${PYTHON_BIN}" evaluate_input_perturbations.py \
      --config_file "${config_file}" \
      --name "${INPUT_METHOD}_w${WINDOW_SIZE}_fold${fold}" \
      --group "${GROUP}" \
      --fold "${fold}" \
      --window_size "${WINDOW_SIZE}" \
      --batch_size "${BATCH_SIZE}" \
      --output_dir "${output_dir}" \
      --perturbations ${PERTURBATIONS}
  done

  run_command "${PYTHON_BIN}" aggregate_input_perturbations.py \
    --input_dir "${output_dir}"
fi
