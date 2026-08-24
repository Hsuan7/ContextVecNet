#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/angle/miniconda3/envs/contextvecnet/bin/python}"
GROUP="${GROUP:-final_preprocessing_v2}"
RESULTS_ROOT="${RESULTS_ROOT:-results/final_preprocessing_v2}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-2}"
PATIENCE="${PATIENCE:-20}"
ACTION="${ACTION:-all}"
SECTIONS="${SECTIONS:-methods window_sizes}"
FOLDS="${FOLDS:-0 1 2 3 4}"
METHODS="${METHODS:-contextvecnet text_bert bert_clip text_clip image_clip concat lstm}"
WINDOW_SIZES="${WINDOW_SIZES:-16 32 64 128}"
WINDOW_SIZE_METHOD="${WINDOW_SIZE_METHOD:-bert_clip}"

section_enabled() {
  [[ " ${SECTIONS} " == *" $1 "* ]]
}

method_enabled() {
  [[ " ${METHODS} " == *" $1 "* ]]
}

config_for_method() {
  case "$1" in
    contextvecnet) echo "configs/combos/multi_only.yaml" ;;
    text_bert) echo "configs/combos/text_only_bert.yaml" ;;
    bert_clip) echo "configs/combos/bert_clip_contextvecnet.yaml" ;;
    text_clip) echo "configs/combos/text_only.yaml" ;;
    image_clip) echo "configs/combos/image_only.yaml" ;;
    concat) echo "configs/combos/text_image_concat.yaml" ;;
    lstm) echo "configs/combos/lstm_only.yaml" ;;
    *) echo "Unknown method: $1" >&2; return 1 ;;
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

train_one() {
  local method="$1"
  local window_size="$2"
  local fold="$3"
  local config_file
  local name
  config_file="$(config_for_method "${method}")"
  name="${method}_w${window_size}_fold${fold}"

  run_command "${PYTHON_BIN}" main_maple.py \
    --config_file "${config_file}" \
    --name "${name}" \
    --group "${GROUP}" \
    --fold "${fold}" \
    --window_size "${window_size}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --early_stopping_patience "${PATIENCE}" \
    --mode dryrun
}

evaluate_one() {
  local method="$1"
  local window_size="$2"
  local fold="$3"
  local section="$4"
  local config_file
  local name
  local output_dir
  local calibrations
  local threshold_strategy
  local extra_evaluation_args
  config_file="$(config_for_method "${method}")"
  name="${method}_w${window_size}_fold${fold}"
  output_dir="${RESULTS_ROOT}/${section}/${method}/w${window_size}"
  if [[ "${method}" == "contextvecnet" && "${section}" == "methods" ]]; then
    calibrations=(none temperature platt)
    threshold_strategy="fixed_0.5"
    extra_evaluation_args=(
      --include_none_validation_f1
      --include_platt_validation_f1
    )
  else
    calibrations=(none)
    threshold_strategy="validation_f1"
    extra_evaluation_args=()
  fi

  run_command "${PYTHON_BIN}" evaluate_calibration_comparison.py \
    --config_file "${config_file}" \
    --name "${name}" \
    --group "${GROUP}" \
    --fold "${fold}" \
    --window_size "${window_size}" \
    --batch_size "${BATCH_SIZE}" \
    --output_dir "${output_dir}" \
    --calibrations "${calibrations[@]}" \
    --threshold_strategy "${threshold_strategy}" \
    "${extra_evaluation_args[@]}"
}

run_one() {
  local method="$1"
  local window_size="$2"
  local fold="$3"
  local section="$4"
  if [[ "${ACTION}" == "train" || "${ACTION}" == "all" ]]; then
    train_one "${method}" "${window_size}" "${fold}"
  fi
  if [[ "${ACTION}" == "evaluate" || "${ACTION}" == "all" ]]; then
    evaluate_one "${method}" "${window_size}" "${fold}" "${section}"
  fi
}

mkdir -p "${RESULTS_ROOT}"

# Method and modality comparison at the paper's primary window size.
if section_enabled "methods"; then
  for fold in ${FOLDS}; do
    for method in ${METHODS}; do
      run_one "${method}" 64 "${fold}" "methods"
    done
  done
fi

# Window-size comparison for the paper's primary multimodal model.
if section_enabled "window_sizes"; then
  for fold in ${FOLDS}; do
    for window_size in ${WINDOW_SIZES}; do
      # Reuse w64 only when this invocation already runs the same method in methods.
      if [[ "${window_size}" == "64" ]] \
        && section_enabled "methods" \
        && method_enabled "${WINDOW_SIZE_METHOD}"; then
        continue
      fi
      run_one "${WINDOW_SIZE_METHOD}" "${window_size}" "${fold}" "window_sizes"
    done
  done
fi

if [[ "${ACTION}" == "evaluate" || "${ACTION}" == "all" ]]; then
  # Include reused w64 files when methods and window sizes run together.
  if section_enabled "methods" \
    && section_enabled "window_sizes" \
    && method_enabled "${WINDOW_SIZE_METHOD}"; then
    mkdir -p "${RESULTS_ROOT}/window_sizes/${WINDOW_SIZE_METHOD}/w64"
  fi
  if [[ "${DRY_RUN:-0}" != "1" ]] \
    && section_enabled "methods" \
    && section_enabled "window_sizes" \
    && method_enabled "${WINDOW_SIZE_METHOD}"; then
    shopt -s nullglob
    window_size_w64_files=(
      "${RESULTS_ROOT}/methods/${WINDOW_SIZE_METHOD}/w64/"*_metrics.csv
      "${RESULTS_ROOT}/methods/${WINDOW_SIZE_METHOD}/w64/"*_predictions.csv
    )
    if ((${#window_size_w64_files[@]})); then
      cp -f \
        "${window_size_w64_files[@]}" \
        "${RESULTS_ROOT}/window_sizes/${WINDOW_SIZE_METHOD}/w64/"
    fi
    shopt -u nullglob
  fi
  run_command "${PYTHON_BIN}" aggregate_final_experiments.py \
    --results_root "${RESULTS_ROOT}"
fi
