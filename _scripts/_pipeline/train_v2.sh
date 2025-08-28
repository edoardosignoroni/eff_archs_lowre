#!/bin/bash
# set -x  # Prints each command before execution
# set -e  # Stops execution on any error

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
if [ ! -d $logs_dir ]; then
    echo "LOGS_DIR not found. Creating LOGS_DIR..."
    mkdir $logs_dir
fi

#check for out dir
if [ ! -d $outputs_dir ]; then
    echo "OUTPUTS_DIR not found. Creating OUTPUTS_DIR..."
    mkdir $outputs_dir
fi

#check for results dir
if [ ! -d $results_dir ]; then
    echo "RESULTS_DIR not found. Creating RESULTS_DIR..."
    mkdir $results_dir
fi

grep -l 'sys.exit(cli_main())\|failed to run command' $logs_dir/*-train.log | while read f; do 
    echo $f
    mv $f $f.error
done

#read tsv to get params. give pattern "^[1-9]" to select all.
grep -E $pattern $scripts_dir/$experiment.csv | while IFS="," read voc tok src tgt model max_updates lr wu_updates dropout label_smoothing max_tokens patience validation_interval; do

    log_file="$logs_dir/$src-$tgt-$voc-$tok-$model-train.log"
    this_dir="$data_dir/$src-$tgt/$voc-$tok"
    save_dir="$models_dir/$src-$tgt/$voc-$tok-$model"

    # Ensure the training is repeated until it completes successfully
    while true; do
        # Check if log file exists and contains the success marker
        if [ -f "$log_file" ]; then
            if grep -q "| INFO | fairseq_cli.train | done training in" "$log_file"; then
                echo "Training already completed successfully for $log_file."
                break
            else
                echo "Training incomplete or failed for $log_file. Restarting..."
                mv "$log_file" "$log_file.error"
            fi
        fi

        echo "source:$src target:$tgt voc_size:$voc tok:$tok model:$model"
        echo "Running training: $log_file"

        #tdetok switch #needed for hft detokenization
        case $tok in
            uni.bse|uni.jnt|uni.nof|uni.jnt.nof|bpe.bse|bpe.jnt|bpe.nof|bpe.jnt.nof)
                tdetok=""
            ;;
            hft.bse|hft.jnt|hft.nof|hft.jnt.nof)
                tdetok="--eval-bleu-detok hftdetok --user-dir /home/pary/src/hftoks/hftfairseq"
            ;;
            *)
                echo "Unknown tokenizer."
                exit
        esac

        #models hyperparams
        IFS='_' read -r encoder_layers decoder_layers embs ffw heads <<< "$model"

        encoder_embed_dim=$embs
        decoder_embed_dim=$embs
        encoder_ffn_embed_dim=$ffw
        decoder_ffn_embed_dim=$ffw
        encoder_attention_heads=$heads
        decoder_attention_heads=$heads

        #start training
        mkdir -p $save_dir
        CUDA_VISIBLE_DEVICES=$gid \
        fairseq-train $this_dir/databin \
        --fp16 \
        --task translation \
        --no-progress-bar \
        --log-interval 500 \
        -s $src \
        -t $tgt \
        --arch transformer \
        --seed 42 \
        --encoder-layers $encoder_layers \
        --decoder-layers $decoder_layers \
        --encoder-embed-dim $encoder_embed_dim \
        --decoder-embed-dim $decoder_embed_dim \
        --encoder-ffn-embed-dim $encoder_ffn_embed_dim \
        --decoder-ffn-embed-dim $decoder_ffn_embed_dim \
        --encoder-attention-heads $encoder_attention_heads \
        --decoder-attention-heads $decoder_attention_heads \
        --share-decoder-input-output-embed \
        --optimizer adam  \
        --adam-betas '(0.9, 0.98)' \
        --clip-norm 0.0 \
        --max-update $max_updates \
        --lr $lr \
        --lr-scheduler inverse_sqrt \
        --warmup-updates $wu_updates \
        --dropout $dropout \
        --weight-decay 0.0001 \
        --criterion label_smoothed_cross_entropy \
        --label-smoothing $label_smoothing \
        --max-tokens $max_tokens \
        --eval-bleu \
        --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
        --eval-bleu-remove-bpe sentencepiece \
        $tdetok \
        --scoring bleu \
        --best-checkpoint-metric bleu \
        --maximize-best-checkpoint-metric \
        --patience $patience \
        --eval-bleu-print-samples \
        --validate-interval $validation_interval \
        --keep-best-checkpoints 2 \
        --keep-last-epochs 2 \
        --save-dir $save_dir &> $log_file
    done
done
