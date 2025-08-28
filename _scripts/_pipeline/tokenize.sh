#!/bin/bash
### CONTROLLER SCRIPT FOR TOKENIZATION
# tokenizes filese according to .tsv experiment files
# and grep pattern

experiment=$1
pattern=$2

project_dir="/nlp/projekty/mtlowre/new_tokeval"
scripts_dir="$project_dir/_scripts"
logs_dir="$project_dir/_logs/$experiment"

#check for log dir
echo "Checking for DIRS..."
if [ ! -d $logs_dir ]; then
    echo "LOGS_DIR not found. Creating LOGS_DIR..."
    mkdir $logs_dir
else
    echo "DIR found."
fi

#check for done/not done logs
echo "Checking for errors and incomplete LOGS..."
if [ -z "$( ls -A $logs_dir )" ]; then
   echo "Empty"
else
   grep -l 'Program terminated with an unrecoverable error.\|command not found\|Keyboard Interrupt' $logs_dir/*-toks.log | while read f; do 
   echo $f
   mv $f $f.error
done
fi

#read tsv to get params
echo "Grepping arguments from experiment .csv..."
grep -E $pattern $scripts_dir/$experiment.csv | while IFS="," read voc tok src tgt model max_updates lr wu_updates dropout label_smoothing max_tokens patience validation_interval; do

    log_file="$logs_dir/$src-$tgt-$voc-$tok-toks.log"

    #tokenize
        if [ ! -f "$log_file" ]; then
            
            echo "source:$src target:$tgt voc_size:$voc tok:$tok"
        
            if [[ $tok == *"uni"* ]]; then
                bash $scripts_dir/tokenize_uni.sh $src $tgt $voc $tok $experiment &> $log_file
            fi
            if [[ $tok == *"bpe"* ]]; then
                bash $scripts_dir/tokenize_bpe.sh $src $tgt $voc $tok $experiment &> $log_file
            fi
            if [[ $tok == *"hft"* ]]; then
                bash $scripts_dir/tokenize_hft.sh $src $tgt $voc $tok $experiment &> $log_file
            fi
        fi
done