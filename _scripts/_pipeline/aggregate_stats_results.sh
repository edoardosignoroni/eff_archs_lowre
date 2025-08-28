#!/bin/bash
experiment=$1

project_dir="/nlp/projekty/mtlowre/new_tokeval"
data_dir="$project_dir/_data"
models_dir="$project_dir/_models/$experiment"
outputs_dir="$project_dir/_outputs/$experiment"
logs_dir="$project_dir/_logs/$experiment"
scripts_dir="$project_dir/_scripts"
results_dir="$project_dir/_results/$experiment"

grep "^[1-9]" $scripts_dir/$experiment.csv | while IFS="," read voc tok src tgt model max_updates lr wu_updates dropout label_smoothing max_tokens patience validation_interval; do
#Aggregate #remove $model from file names

    #check if files exist, if not try reverse direction
    # if [ ! -f "$data_dir/$src-$tgt/train.toks.$tgt" ]; then
    #     tmp_src=$src
    #     src=$tgt
    #     tgt=$tmp_src
    # fi

    stats_file="$outputs_dir/$src-$tgt-$voc-$tok-$model-stats.csv"
    score_file="$outputs_dir/$src-$tgt-$voc-$tok-$model-scr.csv"
    results_file="$outputs_dir/$src-$tgt-$voc-$tok-$model-scrstats.csv"

    rm $results_file

    for file in $outputs_dir/$src-$tgt-$voc-$tok-$model-scr.csv \
                $outputs_dir/$src-$tgt-$voc-$tok.$src-vocstats.csv \
                $outputs_dir/$src-$tgt-$voc-$tok.$tgt-vocstats.csv \
                $outputs_dir/$src-$tgt.$src-train-stats.csv \
                $outputs_dir/$src-$tgt.$src-valid-stats.csv \
                $outputs_dir/$src-$tgt.$src-test-stats.csv \
                $outputs_dir/$src-$tgt.$tgt-train-stats.csv \
                $outputs_dir/$src-$tgt.$tgt-valid-stats.csv \
                $outputs_dir/$src-$tgt.$tgt-test-stats.csv \
                $outputs_dir/$src-$tgt-$voc-$tok.$src-train-tokstats.csv \
                $outputs_dir/$src-$tgt-$voc-$tok.$src-valid-tokstats.csv \
                $outputs_dir/$src-$tgt-$voc-$tok.$src-test-tokstats.csv \
                $outputs_dir/$src-$tgt-$voc-$tok.$tgt-train-tokstats.csv \
                $outputs_dir/$src-$tgt-$voc-$tok.$tgt-valid-tokstats.csv \
                $outputs_dir/$src-$tgt-$voc-$tok.$tgt-test-tokstats.csv; do

        if [ ! -f $f ]; then
            echo "$f" >> "$results_file"
        fi
        
        while IFS= read -r line; do
            echo -n ",$line" | tr -d "()" | tr -d " " >> "$results_file"
        done < "$file"
        sed -e s/^,//g -i $results_file
        # if [ $file != $score_file ]; then
        #     echo "rm $file"
        # fi
    done
done