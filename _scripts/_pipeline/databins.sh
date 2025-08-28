#!/bin/bash
experiment=$1
pattern=$2

project_dir="/nlp/projekty/mtlowre/new_tokeval"
data_dir="$project_dir/_data"
models_dir="$project_dir/_models/$experiment"
outputs_dir="$project_dir/_outputs/$experiment"
logs_dir="$project_dir/_logs/$experiment"
scripts_dir="$project_dir/_scripts"

#check for done/not done logs
grep -l 'Traceback (most recent call last):\|Dictionary: 8 types\|command not found' $logs_dir/*-databin.log | while read f; do 
    echo $f
    mv $f $f.error
done

#read tsv to get params
grep $pattern $scripts_dir/$experiment.csv | while IFS="," read voc tok src tgt model max_updates lr wu_updates dropout label_smoothing max_tokens patience validation_interval; do

    #binarize
    log_file="$logs_dir/$src-$tgt-$voc-$tok-databin.log"
    
    if [ ! -f $log_file ]; then

        echo "source:$src target:$tgt voc_size:$voc tok:$tok"
    
        # create reverse direction
        rev_dir="$data_dir/$tgt-$src"
        mkdir -p $rev_dir

        #setting dest dir and rm previous iteration
        dest_dir=$data_dir/$src-$tgt/$voc-$tok
        rm -rf $dest_dir/databin

        if [ -f "$dest_dir/test.toks.$src" ] && [ -f "$dest_dir/test.toks.$tgt" ]; then
            echo "Test files exist. Binarizing..." &> $log_file
            fairseq-preprocess \
            --source-lang $src \
            --target-lang $tgt \
            --trainpref $dest_dir/train.toks \
            --validpref $dest_dir/valid.toks \
            --testpref $dest_dir/test.toks \
            --destdir $dest_dir/databin \
            --workers 20 &>> $log_file

        else
            echo "Test files DO NOT exist. Binarizing train and valid..." &> $log_file
            fairseq-preprocess \
            --source-lang $src \
            --target-lang $tgt \
            --trainpref $dest_dir/train.toks \
            --validpref $dest_dir/valid.toks \
            --destdir $dest_dir/databin \
            --workers 20 &>> $log_file
        fi

                # creates symbolic links for reverse direction
                rm -rf $rev_dir/$voc-$tok/databin
                mkdir -p $rev_dir/$voc-$tok ; ln -s $data_dir/$src-$tgt/$voc-$tok/databin $rev_dir/$voc-$tok
    fi
    echo "DONE DONE DONE" &>> $log_file
done