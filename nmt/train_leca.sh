export CUDA_VISIBLE_DEVICES=0,1

name=$1
gpu_num=2
DATA_PATH=./data_bin/$name
MODEL_PATH=checkpoints/$name
LOG_PATH=log/$name.log

mkdir -p checkpoints/$name
mkdir -p $MODEL_PATH

nohup fairseq-train $DATA_PATH \
    --arch transformer_leca --task translation_leca --user-dir $user_dir \
    --dropout 0.3 --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --clip-norm 0.0 --lr 7e-4 --lr-scheduler inverse_sqrt \
    --warmup-updates 2000 --warmup-init-lr 1e-07 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5}' \
    --eval-tokenized-bleu \
    --eval-bleu-remove-bpe \
    --best-checkpoint-metric bleu \
    --maximize-best-checkpoint-metric \
    --max-tokens 4096 --max-epoch 20 --save-interval-updates 500 \
    --distributed-world-size $gpu_num --update-freq 4 \
    --keep-interval-updates 10 --keep-last-epochs 10 \
    --share-all-embeddings --share-decoder-input-output-embed \
    --log-interval 10 --save-dir $MODEL_PATH  --fp16 >> $LOG_PATH 2>&1 &

