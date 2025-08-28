#!/bin/bash

#arguments
experiment=$1
pattern=$2
#toks: hft hft.jnt hft.nof hft.jnt.nof bpe bpe.jnt bpe.nof bpe.jnt.nof uni uni.jnt uni.nof uni.jnt.nof
#models: base small

#setting the directories
project_dir="/nlp/projekty/mtlowre/new_tokeval"
data_dir="$project_dir/_data"
models_dir="$project_dir/_models/$experiment"
outputs_dir="$project_dir/_outputs/$experiment"
logs_dir="$project_dir/_logs/$experiment"
scripts_dir="$project_dir/_scripts"
results_dir="$project_dir/_results/$experiment"

#check for log dir
# echo "Checking for LOGS_DIR..."
if [ ! -d $logs_dir ]; then
    echo "LOGS_DIR not found. Creating LOGS_DIR..."
    mkdir $logs_dir
fi

#check for out dir
# echo "Checking for OUTPUTS_DIR..."
if [ ! -d $outputs_dir ]; then
    echo "OUTPUTS_DIR not found. Creating OUTPUTS_DIR..."
    mkdir $outputs_dir
fi

#check for out dir
# echo "Checking for RESULTS_DIR..."
if [ ! -d $results_dir ]; then
    echo "RESULTS_DIR not found. Creating RESULTS_DIR..."
    mkdir $results_dir
fi

echo "Grepping errors"
grep -l "Traceback (most recent call last):\|TypeError" $logs_dir/*-eval.log | while read f; do 
    echo $f
    mv $f $f.error
done

echo "Starting evaluation"
#read tsv to get params. give pattern "^[1-9]" to select all.
#genpairs.tsv = trainpairs.csv, just for testing. It will read from trainpairs.tsv and do everything together in the next iteration
grep -E $pattern $project_dir/_scripts/$experiment.csv | while IFS="," read voc tok src tgt model max_updates lr wu_updates dropout label_smoothing max_tokens patience validation_interval; do
    #train

    if [ ! -f "$logs_dir/$src-$tgt-$voc-$tok-$model-eval.log" ]; then

        echo "source:$src target:$tgt voc_size:$voc tok:$tok model:$model"

        log_file="$logs_dir/$src-$tgt-$voc-$tok-$model-eval.log"

        python3 $project_dir/_scripts/eval.py \
            --source_lang $src \
            --target_lang $tgt \
            --vocab_size $voc \
            --tokenizer $tok \
            --model $model \
            --experiment $experiment \
            --bootstrap_eval &> $log_file
    fi
done