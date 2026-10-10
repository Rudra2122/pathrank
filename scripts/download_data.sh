#!/usr/bin/env bash
# Download the CareerBuilder Job Recommendation Challenge data from Kaggle.
# 1) pip install kaggle; put your API token at ~/.kaggle/kaggle.json (kaggle.com -> Settings -> API)
# 2) Accept the competition rules once: https://www.kaggle.com/c/job-recommendation/rules
set -euo pipefail
OUT=data/raw/careerbuilder
mkdir -p "$OUT"
kaggle competitions download -c job-recommendation -p "$OUT"
cd "$OUT"
# the competition zip contains per-file zips (apps.tsv.zip, jobs.zip, ...): unzip until none left
for _ in 1 2 3; do
  shopt -s nullglob
  zips=(*.zip)
  [ ${#zips[@]} -eq 0 ] && break
  for z in "${zips[@]}"; do unzip -o -q "$z" && rm "$z"; done
done
echo "files:"; ls -lh
for f in apps.tsv users.tsv jobs.tsv user_history.tsv; do
  p=$(find . -name "$f" | head -1)
  [ -n "$p" ] && echo "$f: $(($(wc -l < "$p") - 1)) rows" || echo "MISSING $f"
done
