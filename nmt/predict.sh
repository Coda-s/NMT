export CUDA_VISIBLE_DEVICES=0

predict(){
    model_path=./checkpoints/$name
    data_path=./data_bin/$name
    result_path=./result/$name

    mkdir -p $result_path
    fairseq-generate $data_path \
        --path $model_path/checkpoint_best.pt \
        --batch-size 128 \
        --remove-bpe --sacrebleu \
        --beam 5 >> $path/test.out
    grep ^H $path/test.out | sort -n -k 2 -t '-' | cut -f 3 > $path/test.prediction
    grep ^T $path/test.out | sort -n -k 2 -t '-' | cut -f 2 > $path/test.tgt
    grep ^S $path/test.out | sort -n -k 2 -t '-' | cut -f 2 > $path/test.src
}

name=$1
if [ $name = 'labeled_3' -o $name = 'labeled_4' ];then
    python post_process.py --name result/$name
fi
# predict $name

# without post_process
# python evaluate.py --model $name
# with post_process
# python evaluate.py --model $name --clean

