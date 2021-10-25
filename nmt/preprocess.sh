moses_scripts=../nmt_work/utils/mosesdecoder/scripts # mose path
bpe_scripts=../nmt_work/utils/subword-nmt # subword-nmt path

tokenize(){
    data_path=$1
    split=$2
    S=$3
    T=$4
    perl $moses_scripts/tokenizer/tokenizer.perl -threads 50 -l $S < $data_path/$split.raw.$S > $data_path/$split.tok.$S
    perl $moses_scripts/tokenizer/tokenizer.perl -threads 50 -l $T < $data_path/$split.raw.$T > $data_path/$split.tok.$T
}

learn_bpe(){
    bpe_operations=20000
    data_path=$1
    S=$2
    T=$3
    python $bpe_scripts/learn_bpe.py -s $bpe_operations < $data_path/train.tok.$S > $data_path/bpe.code.$S
    python $bpe_scripts/learn_bpe.py -s $bpe_operations < $data_path/train.tok.$T > $data_path/bpe.code.$T
}

match(){
    file_path=$1
    save_path=$2
    fdict=$1/vocab.txt
    batch_num=$3
    split=$4
    S=$5
    T=$6
    
    python match.py \
        --file_path $file_path --save_path $save_path \
        --fdict $fdict --s $S --t $T \
        --split $split --batch_num $batch_num

    for i in `seq $batch_num`
    do  
        rm $data_path/$split$i.$S
        rm $data_path/$split$i.$T
        rm $save_path/$split$i.$S.matched
        rm $save_path/$split$i.$T.matched
        rm $save_path/$split$i.term_lines
    done
}

tag(){
    match_path=$1
    output_path=$2
    split=$3
    S=$4
    T=$5
    python tag.py \
        --match_path $match_path \
        --output_path $output_path \
        --split $split --slang $S --tlang $T
}

apply_bpe(){
    path=$1
    bpe_path=$2
    split=$3
    S=$4
    T=$5
    python $bpe_scripts/apply_bpe.py -c $bpe_path/bpe.code.$S < $path/${split}.tok.$S > $path/${split}.bpe.$S
    python $bpe_scripts/apply_bpe.py -c $bpe_path/bpe.code.$T < $path/${split}.tok.$T > $path/${split}.bpe.$T
    python merge_label.py --input_file $path/${split}.bpe.$S --output_file $path/${split}.bpe.merge.$S
    python merge_label.py --input_file $path/${split}.bpe.$T --output_file $path/${split}.bpe.merge.$T
}

binary(){
    mkdir -p $2
    S=$3
    T=$4
    fairseq-preprocess -s $S -t $T \
        --trainpref $1/train.bpe.merge \
        --validpref $1/valid.bpe.merge \
        --testpref $1/test.bpe.merge \
        --joined-dictionary \
        --workers 20 \
        --destdir $2
}

data_path=data
match_path=data_match
label_path=data_label
bin_path=data_bin
mkdir -p $match_path $label_path $bin_path
S=en
T=de
splits=("train" "valid" "test")

echo "tokenize"
for split in ${splits[@]}
do
    tokenize $data_path $split $S $T
done

echo "learn_bpe"
learn_bpe $data_path $S $T

echo "match..."
rm -rf $match_path/*
match $data_path $match_path 5 train $S $T
match $data_path $match_path 5 valid $S $T
cp $data_path/test.tok.$S $match_path/test.matched.$S
cp $data_path/test.tok.$T $match_path/test.matched.$T
cp $data_path/test.term_lines $match_path
echo "done"


echo "tag..."
mkdir -p $label_path/original
for split in ${splits[@]}
do 
    tag $match_path $label_path $split $S $T
    for i in {1..7}
    do
        if [ $split = 'test' ]; then
            cat $label_path/labeled_$i/$split.label.$S > $label_path/labeled_$i/$split.tok.$S
            cat $label_path/labeled_$i/$split.label.$T > $label_path/labeled_$i/$split.tok.$T
        else
            cat $data_path/$split.tok.$S $label_path/labeled_$i/$split.label.$S > $label_path/labeled_$i/$split.tok.$S
            cat $data_path/$split.tok.$T $label_path/labeled_$i/$split.label.$T > $label_path/labeled_$i/$split.tok.$S
        fi
    done
    cp $data_path/$split.tok.* $label_path/original
done
echo "done"


echo "apply_bpe..."
for split in ${splits[@]}
do
    apply_bpe $label_path/original $data_path $split $S $T
    for i in {1..7}
    do
        apply_bpe $label_path/labeled_$i $data_path $split $S $T
    done
done
echo "done..."

echo "binary..."
binary $label_path/labeled_$i $bin_path/original $S $T
for i in {1..7}
do
    binary $label_path/labeled_$i $bin_path/labeled_$i $S $T
done
echo "done..."
