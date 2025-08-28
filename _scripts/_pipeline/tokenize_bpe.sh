#!/bin/bash

min_freq=100
src=$1
tgt=$2
vocs=$3
toks=$4
experiment=$5
spm=/nlp/projekty/mtlowre/fairseq/lib

project_dir="/nlp/projekty/mtlowre/new_tokeval"
data_dir="$project_dir/_data/$src-$tgt"
models_dir="$project_dir/_models/$experiment"
outputs_dir="$project_dir/_outputs/$experiment"
logs_dir="$project_dir/_logs/$experiment"

# joint file
cat $data_dir/train.$src $data_dir/train.$tgt > $data_dir/train.jnt

for voc in $vocs; do
    for tok in $toks; do
        #character coverage switch
        cover=1.0
        if [[ $voc == 250 || $voc == 500 ]]; then
            cover=0.98
        fi
        
        echo "Creating $voc-$tok..."
        dest_dir=$data_dir/$voc-$tok
       
        mkdir -p $dest_dir

        if [ -f "$data_dir/test.$src" ] && [ -f "$data_dir/test.$tgt" ]; then
            echo "Test files exist. Preprocessing..."
            splits="train valid test"
        else
            echo "Test files DO NOT exist. Setting splits to "train valid"..."
            splits="train valid"
        fi
        

        case $tok in
            bpe.bse)
            #	train
                LD_LIBRARY_PATH=$spm spm_train --input=$data_dir/train.$src --model_prefix=$dest_dir/$src --vocab_size=$voc --character_coverage=$cover --model_type=bpe --split_by_whitespace=true
                LD_LIBRARY_PATH=$spm spm_train --input=$data_dir/train.$tgt --model_prefix=$dest_dir/$tgt --vocab_size=$voc --character_coverage=$cover --model_type=bpe --split_by_whitespace=true
                LD_LIBRARY_PATH=$spm spm_encode --model=$dest_dir/$src.model --generate_vocabulary < $data_dir/train.$src > $dest_dir/$src.vocab
                LD_LIBRARY_PATH=$spm spm_encode --model=$dest_dir/$tgt.model --generate_vocabulary < $data_dir/train.$tgt > $dest_dir/$tgt.vocab
            #	tokenize
                for split in $splits; do
                LD_LIBRARY_PATH=$spm spm_encode --model=$dest_dir/$src.model --output_format=piece --vocabulary=$dest_dir/$src.vocab --vocabulary_threshold=$min_freq < $data_dir/$split.$src  > $dest_dir/$split.toks.$src
                LD_LIBRARY_PATH=$spm spm_encode --model=$dest_dir/$tgt.model --output_format=piece --vocabulary=$dest_dir/$tgt.vocab --vocabulary_threshold=$min_freq < $data_dir/$split.$tgt  > $dest_dir/$split.toks.$tgt
                done
            ;;
            bpe.jnt)
                #	train
                LD_LIBRARY_PATH=$spm spm_train --input=$data_dir/train.jnt --model_prefix=$dest_dir/jnt --vocab_size=$voc --character_coverage=$cover --model_type=bpe --split_by_whitespace=true
                LD_LIBRARY_PATH=$spm spm_encode --model=$dest_dir/jnt.model --generate_vocabulary < $data_dir/train.$src > $dest_dir/$src.jnt.vocab
                LD_LIBRARY_PATH=$spm spm_encode --model=$dest_dir/jnt.model --generate_vocabulary < $data_dir/train.$tgt > $dest_dir/$tgt.jnt.vocab
                #	tokenize
                for split in $splits; do
                LD_LIBRARY_PATH=$spm spm_encode --model=$dest_dir/jnt.model --vocabulary=$dest_dir/$src.jnt.vocab --output_format=piece --vocabulary_threshold=$min_freq < $data_dir/$split.$src  > $dest_dir/$split.toks.$src
                LD_LIBRARY_PATH=$spm spm_encode --model=$dest_dir/jnt.model --vocabulary=$dest_dir/$tgt.jnt.vocab --output_format=piece --vocabulary_threshold=$min_freq < $data_dir/$split.$tgt  > $dest_dir/$split.toks.$tgt
                done
            ;;
            bpe.nof)
            #	train
                LD_LIBRARY_PATH=$spm spm_train --input=$data_dir/train.$src --model_prefix=$dest_dir/$src --vocab_size=$voc --character_coverage=$cover --model_type=bpe --split_by_whitespace=true
                LD_LIBRARY_PATH=$spm spm_train --input=$data_dir/train.$tgt --model_prefix=$dest_dir/$tgt --vocab_size=$voc --character_coverage=$cover --model_type=bpe --split_by_whitespace=true
            #	tokenize
                for split in $splits; do
                LD_LIBRARY_PATH=$spm spm_encode --model=$dest_dir/$src.model --output_format=piece < $data_dir/$split.$src  > $dest_dir/$split.toks.$src
                LD_LIBRARY_PATH=$spm spm_encode --model=$dest_dir/$tgt.model --output_format=piece < $data_dir/$split.$tgt  > $dest_dir/$split.toks.$tgt
                done
            ;;
            bpe.jnt.nof)
            #	train
                LD_LIBRARY_PATH=$spm spm_train --input=$data_dir/train.jnt --model_prefix=$dest_dir/jnt --vocab_size=$voc --character_coverage=$cover --model_type=bpe --split_by_whitespace=true
                LD_LIBRARY_PATH=$spm spm_encode --model=$dest_dir/jnt.model --generate_vocabulary < $data_dir/train.$src > $dest_dir/$src.jnt.vocab
                LD_LIBRARY_PATH=$spm spm_encode --model=$dest_dir/jnt.model --generate_vocabulary < $data_dir/train.$tgt > $dest_dir/$tgt.jnt.vocab
            #	tokenize
                for split in $splits; do
                LD_LIBRARY_PATH=$spm spm_encode --model=$dest_dir/jnt.model --vocabulary=$dest_dir/$src.jnt.vocab --output_format=piece  < $data_dir/$split.$src  > $dest_dir/$split.toks.$src
                LD_LIBRARY_PATH=$spm spm_encode --model=$dest_dir/jnt.model --vocabulary=$dest_dir/$tgt.jnt.vocab --output_format=piece  < $data_dir/$split.$tgt  > $dest_dir/$split.toks.$tgt
                done
            ;;
            *)
                echo $tok not found
            ;;
        esac
    done
done