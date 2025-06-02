#!/usr/bin/env bash

# ----------------------------------------------------------------
# run_all.sh
#
# Runs generate_kg.py → generate_pg.py → generate_tps.py in order,
# suppressing all output, and prints the elapsed time for each step
# as well as the total elapsed time.
# ----------------------------------------------------------------

# Exit immediately if any command fails
set -e

# Record overall start time
overall_start=$(date +%s)

# 1) Run generate_kg.py
step1_start=$(date +%s)
if [ -f "./generate_kg.py" ]; then
  python3 generate_kg.py > /dev/null 2>&1
else
  echo "Error: generate_kg.py not found." >&2
  exit 1
fi
step1_end=$(date +%s)
step1_elapsed=$((step1_end - step1_start))
printf "Step 1 (generate_kg.py) elapsed time: %02d:%02d:%02d\n" \
       $((step1_elapsed/3600)) $(((step1_elapsed%3600)/60)) $((step1_elapsed%60))

# 2) Run generate_pg.py
step2_start=$(date +%s)
if [ -f "./generate_pg.py" ]; then
  python3 generate_pg.py > /dev/null 2>&1
else
  echo "Error: generate_pg.py not found." >&2
  exit 1
fi
step2_end=$(date +%s)
step2_elapsed=$((step2_end - step2_start))
printf "Step 2 (generate_pg.py) elapsed time: %02d:%02d:%02d\n" \
       $((step2_elapsed/3600)) $(((step2_elapsed%3600)/60)) $((step2_elapsed%60))

# 3) Run generate_tps.py
step3_start=$(date +%s)
if [ -f "./generate_tps.py" ]; then
  python3 generate_tps.py > /dev/null 2>&1
else
  echo "Error: generate_tps.py not found." >&2
  exit 1
fi
step3_end=$(date +%s)
step3_elapsed=$((step3_end - step3_start))
printf "Step 3 (generate_tps.py) elapsed time: %02d:%02d:%02d\n" \
       $((step3_elapsed/3600)) $(((step3_elapsed%3600)/60)) $((step3_elapsed%60))

# Compute and print overall elapsed time
overall_end=$(date +%s)
overall_elapsed=$((overall_end - overall_start))
printf "Total elapsed time:              %02d:%02d:%02d\n" \
       $((overall_elapsed/3600)) $(((overall_elapsed%3600)/60)) $((overall_elapsed%60))
