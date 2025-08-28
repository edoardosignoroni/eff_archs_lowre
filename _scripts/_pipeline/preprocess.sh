#!/bin/bash
# Controller script for fairseq preprocessing pipeline
# Preprocesses data for fairseq training, 
# according to .tsv experiment file and grep pattern

experiment=$1
pattern=$2

project_dir="/nlp/projekty/mtlowre/new_tokeval"
scripts_dir="$project_dir/_scripts"

echo "Starting tokenization at $(date +"%Y%m%d_%H%M%S")"
bash $scripts_dir/tokenize.sh $experiment $pattern
echo "Starting databins at $(date +"%Y%m%d_%H%M%S")"
bash $scripts_dir/databins.sh $experiment $pattern
echo "Done preprocessing at $(date +"%Y%m%d_%H%M%S")"
