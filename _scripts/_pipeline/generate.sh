#!/bin/bash

#arguments
gid=$1
experiment=$2
pattern=$3

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
grep -l 'Traceback (most recent call last):\|command not found' $logs_dir/*-gen.log | while read f; do 
    echo $f
    mv $f $f.error
done

echo "Starting generation"
#read tsv to get params. give pattern "^[1-9]" to select all.
#grep $pattern $scripts_dir/trainpairs.tsv | while read voc tok model src tgt; do
#genpairs.tsv = trainpairs.csv, just for testing. It will read from trainpairs.tsv and do everything together in the next iteration
grep -E $pattern $scripts_dir/$experiment.csv | while IFS="," read voc tok src tgt model max_updates lr wu_updates dropout label_smoothing max_tokens patience validation_interval; do

    #generate
    log_file="$logs_dir/$src-$tgt-$voc-$tok-$model-gen.log"

    if [ ! -f "$log_file" ]; then

        echo "source:$src target:$tgt voc_size:$voc tok:$tok model:$model"
        
        #detok switch
        case $tok in
            uni.bse|uni.jnt|uni.nof|uni.jnt.nof|bpe.bse|bpe.jnt|bpe.nof|bpe.jnt.nof)
                detok="--remove-bpe sentencepiece"
            ;;
            hft.bse|hft.jnt|hft.nof|hft.jnt.nof)
                detok="--user-dir /home/pary/src/hftoks/hftfairseq --tokenizer hftdetok"
            ;;
            *)
                echo "Unknown tokenizer"
                exit
        esac

        output_file="$outputs_dir/$src-$tgt-$voc-$tok-$model.gen"

        databin="$data_dir/$src-$tgt/$voc-$tok/databin"

        CUDA_VISIBLE_DEVICES=$gid \
        fairseq-generate "$databin" -s $src -t $tgt \
        --batch-size 1 \
        --path "$models_dir/$src-$tgt/$voc-$tok-$model/checkpoint_best.pt" \
        $detok \
        --scoring sacrebleu &> $log_file
        
        #copy log to output file in lang pair directory for next steps
        cp $log_file $output_file

        #grep source side
        source_file="$outputs_dir/$src-$tgt.$src"
        grep ^S- $output_file | sed -e "s/^S-//" | sort -n | cut -f2 > $source_file
        #grep reference side 
        references_file="$outputs_dir/$src-$tgt.$tgt"
        grep ^T- $output_file | sed -e "s/^T-//" | sort -n | cut -f2 > $references_file
        #grep translations
        translations_file="$outputs_dir/$src-$tgt-$voc-$tok-$model.$tgt"
        grep ^D- $output_file | sed -e "s/^D-//" | sort -n | cut -f3 > $translations_file

    fi
done    