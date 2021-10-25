Moses_dir=/data/zhanghongxiao/tools/MosesSMT/mosesdecoder
data_dir=/data/zhanghongxiao/NMT/en-de/dataset
work_dir=/data/zhanghongxiao/NMT/en-de/SMT
dict_dir=/data/zhanghongxiao/NMT/en-de/dict/smt_dict

# 1.tok & tc & bpe
sh tok_tc_bpe.sh $Moses_dir $data_dir $work_dir

# 2.train
sh train.sh $Moses_dir $work_dir

# 3.table to vocab
cd $work_dir/working/train/model
gunzip phrase-table.gz
cp phrase-table $work_dir/phrase/
cd $work_dir/phrase/
sh table_to_vocab.sh $work_dir $dict_dir