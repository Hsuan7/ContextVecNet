#!/usr/bin/env bash
set -euo pipefail
cd /home/angle/ContextVecNet
export PATH=/home/angle/miniconda3/envs/contextvecnet/bin:$PATH
while IFS= read -r cmd; do
  echo "RUNNING: $cmd"
  eval "$cmd"
done < interpretability_outputs/bert_clip_w64_main_xai/extract_attention_commands.txt