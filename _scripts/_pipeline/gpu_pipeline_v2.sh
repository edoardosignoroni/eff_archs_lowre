#!/bin/bash

# set -x  # Prints each command before execution
# set -e  # Stops execution on any error

gid=$1
experiment=$2
pattern=$3

project_dir="/nlp/projekty/mtlowre/new_tokeval"
scripts_dir="$project_dir/_scripts"

echo "Starting training at $(date +"%Y%m%d_%H%M%S")"
bash $scripts_dir/train_v2.sh $gid $experiment $pattern
echo "Starting generation at $(date +"%Y%m%d_%H%M%S")"
bash $scripts_dir/generate.sh $gid $experiment $pattern
echo "Starting evaluation at $(date +"%Y%m%d_%H%M%S")"
bash $scripts_dir/eval.sh $experiment $pattern
echo "Collecting stats..."
# bash $scripts_dir/data_stats.sh $experiment $pattern
# echo "Aggregating scores and statistics at $(date +"%Y%m%d_%H%M%S")"
# bash $scripts_dir/aggregate_stats_results.sh $experiment
echo "Collecting results at $(date +"%Y%m%d_%H%M%S")"
bash $scripts_dir/all_results.sh $experiment
# echo "Gathering additional data at $(date +"%Y%m%d_%H%M%S")"
# bash $scripts_dir/train_times.sh
echo "Finished at $(date +"%Y%m%d_%H%M%S")"
wait