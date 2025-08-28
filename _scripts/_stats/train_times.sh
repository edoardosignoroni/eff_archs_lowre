#!/bin/bash
project_dir="/nlp/projekty/mtlowre/new_tokeval"
scripts_dir="$project_dir/_scripts"
logs_dir="$project_dir/_logs"
results_dir="$project_dir/_results"

results_file="$results_dir/times-$(date +"%Y%m%d_%H%M%S").csv"

train_logs=$(ls -f $logs_dir | grep train)

for f in $train_logs; do
    src=$(echo $f | sed -e "s/-train.log$//" | cut -d '-' -f 1)
    tgt=$(echo $f | sed -e "s/-train.log$//" | cut -d '-' -f 2)
    voc=$(echo $f | sed -e "s/-train.log$//" | cut -d '-' -f 3)
    tok=$(echo $f | sed -e "s/-train.log$//" | cut -d '-' -f 4)
    model=$(echo $f | sed -e "s/-train.log$//" | cut -d '-' -f 5)
    params=$(grep 'num. shared model params:' $logs_dir/$f | cut -d ' ' -f 12)
    total_train_time=$(grep 'done training in' $logs_dir/$f | cut -d ' ' -f 11)
    avg_train_epoch=$(python $scripts_dir/get_train_times.py --log $logs_dir/$f)
    if grep -q 'early stop since valid performance' "$logs_dir/$f"; then
        end_condition='early_stop'
    else
        end_condition='max_updates'
    fi
    
    printf "$src\t$tgt\t$voc\t$tok\t$model\t$params\t$total_train_time\t$avg_train_epoch\t$end_condition\n" >> $results_file
done