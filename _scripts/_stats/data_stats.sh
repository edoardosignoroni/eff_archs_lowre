#!/bin/bash
# Controller script for collecting stats about the dataset
# according to .tsv experiment file and grep pattern

experiment=$1
pattern=$2

project_dir="/nlp/projekty/mtlowre/new_tokeval"
data_dir="$project_dir/_data"
models_dir="$project_dir/_models/$experiment"
outputs_dir="$project_dir/_outputs/$experiment"
logs_dir="$project_dir/_logs/$experiment"
scripts_dir="$project_dir/_scripts"
results_dir="$project_dir/_results/$experiment"

#check for log dir
echo "Checking for LOGS_DIR..."
if [ ! -d $logs_dir ]; then
    echo "LOGS_DIR not found. Creating LOGS_DIR..."
    mkdir $logs_dir
else
    echo "LOGS_DIR found."
fi

#check for out dir
echo "Checking for OUTPUTS_DIR..."
if [ ! -d $outputs_dir ]; then
    echo "OUTPUTS_DIR not found. Creating OUTPUTS_DIR..."
    mkdir $outputs_dir
else
    echo "OUTPUTS_DIR found."
fi

#check for out dir
echo "Checking for RESULTS_DIR..."
if [ ! -d $results_dir ]; then
    echo "RESULTS_DIR not found. Creating RESULTS_DIR..."
    mkdir $results_dir
else
    echo "RESULTS_DIR found."
fi

#check for done/not done logs
grep -l 'Traceback (most recent call last):\|command not found' $logs_dir/*-stats.log | while read f; do 
    echo $f
    mv $f $f.error
done

grep -L DONE $logs_dir/*-stats.log | while read f; do
    echo $f
    mv $f $f.error
done

#read tsv to get params
grep $pattern $scripts_dir/$experiment.csv | while IFS="," read voc tok src tgt model max_updates lr wu_updates dropout label_smoothing max_tokens patience validation_interval; do

    #collect stats
    log_file="$logs_dir/$src-$tgt-$voc-$tok-stats.log"

    data_src=$src
    data_tgt=$tgt

    # check if files exist, if not try reverse direction
    if [ ! -f "$data_dir/$src-$tgt/train.$tgt" ]; then
        data_src=$tgt
        data_tgt=$src
    fi

    # if [ ! -f $log_file ]; then
    if [ 1 == 1 ]; then # debug
        
        #Collect untokenized data
        echo "Collecting untokenized data stats..." &> $log_file
        python $scripts_dir/data_stats.py -f $data_dir/$data_src-$data_tgt/train.$src > $outputs_dir/$src-$tgt.$src-train-stats.csv
        python $scripts_dir/data_stats.py -f $data_dir/$data_src-$data_tgt/train.$tgt > $outputs_dir/$src-$tgt.$tgt-train-stats.csv
        python $scripts_dir/data_stats.py -f $data_dir/$data_src-$data_tgt/valid.$src > $outputs_dir/$src-$tgt.$src-valid-stats.csv
        python $scripts_dir/data_stats.py -f $data_dir/$data_src-$data_tgt/valid.$tgt > $outputs_dir/$src-$tgt.$tgt-valid-stats.csv
        python $scripts_dir/data_stats.py -f $data_dir/$data_src-$data_tgt/test.$src > $outputs_dir/$src-$tgt.$src-test-stats.csv
        python $scripts_dir/data_stats.py -f $data_dir/$data_src-$data_tgt/test.$tgt > $outputs_dir/$src-$tgt.$tgt-test-stats.csv
        
        #Collect tokenized data
        echo "Collecting tokenized data stats..." &>> $log_file
        python $scripts_dir/tok_stats.py -f $data_dir/$data_src-$data_tgt/$voc-$tok/train.toks.$src -d $data_dir/$src-$tgt/$voc-$tok/databin/dict.$src.txt -t $tok -src $src -tgt $tgt -e $experiment > $outputs_dir/$src-$tgt-$voc-$tok.$src-train-tokstats.csv
        python $scripts_dir/tok_stats.py -f $data_dir/$data_src-$data_tgt/$voc-$tok/train.toks.$tgt -d $data_dir/$src-$tgt/$voc-$tok/databin/dict.$tgt.txt -t $tok -src $src -tgt $tgt -e $experiment > $outputs_dir/$src-$tgt-$voc-$tok.$tgt-train-tokstats.csv
        python $scripts_dir/tok_stats.py -f $data_dir/$data_src-$data_tgt/$voc-$tok/valid.toks.$src -d $data_dir/$src-$tgt/$voc-$tok/databin/dict.$src.txt -t $tok -src $src -tgt $tgt -e $experiment > $outputs_dir/$src-$tgt-$voc-$tok.$src-valid-tokstats.csv
        python $scripts_dir/tok_stats.py -f $data_dir/$data_src-$data_tgt/$voc-$tok/valid.toks.$tgt -d $data_dir/$src-$tgt/$voc-$tok/databin/dict.$tgt.txt -t $tok -src $src -tgt $tgt -e $experiment > $outputs_dir/$src-$tgt-$voc-$tok.$tgt-valid-tokstats.csv
        python $scripts_dir/tok_stats.py -f $data_dir/$data_src-$data_tgt/$voc-$tok/test.toks.$src -d $data_dir/$src-$tgt/$voc-$tok/databin/dict.$src.txt -t $tok -src $src -tgt $tgt -e $experiment > $outputs_dir/$src-$tgt-$voc-$tok.$src-test-tokstats.csv
        python $scripts_dir/tok_stats.py -f $data_dir/$data_src-$data_tgt/$voc-$tok/test.toks.$tgt -d $data_dir/$src-$tgt/$voc-$tok/databin/dict.$tgt.txt -t $tok -src $src -tgt $tgt -e $experiment > $outputs_dir/$src-$tgt-$voc-$tok.$tgt-test-tokstats.csv   
    fi

    echo "DONE" &>> $log_file
done

