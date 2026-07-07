#!/usr/bin/env bash
set -u

cd /home/daniel/Desktop/Integrative-Genomic-Knowledge-Framework-IGKF-main || exit 1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gwas-env

manifest="evaluation/outputs/batch_metadata/gpt_generation_1783433892_manifest.json"

while true; do
  date -Is
  if ! python -m evaluation.openai_batch status "$manifest"; then
    echo "status_check_failed"
    sleep 300
    continue
  fi

  status=$(jq -r ".batch_status.status" "$manifest")
  echo "status=$status"

  if [ "$status" = "completed" ]; then
    python -m evaluation.openai_batch download "$manifest"
    break
  fi

  if [ "$status" = "failed" ] || [ "$status" = "expired" ] || [ "$status" = "cancelled" ]; then
    break
  fi

  sleep 300
done
