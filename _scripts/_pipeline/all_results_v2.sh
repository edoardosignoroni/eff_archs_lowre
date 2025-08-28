#!/bin/bash
experiment=$1

project_dir="/nlp/projekty/mtlowre/new_tokeval"
data_dir="$project_dir/_data"
models_dir="$project_dir/_models/$experiment"
outputs_dir="$project_dir/_outputs/$experiment"
logs_dir="$project_dir/_logs/$experiment"
scripts_dir="$project_dir/_scripts"
results_dir="$project_dir/_results/$experiment"

results_file="$results_dir/results-$(date +"%Y%m%d_%H%M%S").csv"

echo 'src,tgt,voc,tok,model,bleu,low_conf_bleu,high_conf_bleu,chrf,low_conf_chrf,high_conf_chrf,chrf++,low_conf_chrf++,high_conf_chrf++,comet,low_conf_comet,high_conf_comet,norm_comet,enc_layers,dec_layers,embs,ffw' > $results_file

ls $outputs_dir/*-scr.csv | while read f; do
    # echo $f
    # printf "\n" &>> $results_file
    # check and purge wrong .scrstats
    if awk -F',' '$1 !~ /^[0-9]+$/ {exit 1}' "$f"; then
        echo $f
        rm $f
    else
        awk 1 $f  &>> $results_file
    fi
done
