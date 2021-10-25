mkdir -p ./lm
SCRIPTS=$1/scripts
BIN=$1/bin
TOOLS=$1/tools
LMPLZ=$BIN/lmplz
BuILD_BINARY=$BIN/build_binary
QUERY=$BIN/query
TRAIN_PERL=$SCRIPTS/training/train-model.perl

data_dir=$2/data_deen
model_dir=$2/lm
work_dir=$2/working/train
mkdir -p $work_dir
result_path=$work_dir/training.out

cd ./lm

$LMPLZ -o 3 < $data_dir/train.clean.de > $model_dir/arpa.de
$BuILD_BINARY $model_dir/arpa.de $model_dir/blm.de



nohup nice $TRAIN_PERL -root-dir $work_dir \
    -corpus $data_dir/train.clean \
    -f en -e de \
    -alignment grow-diag-final-and \
    -reordering msd-bidirectional-fe \
    -lm 0:3:$model_dir/blm.de:8 \
    -external-bin-dir $TOOLS \
    -cores 8 \
    > $result_path 2>&1  &