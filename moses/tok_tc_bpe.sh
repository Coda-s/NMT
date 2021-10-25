SCRIPTS=$1/scripts
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
TRAIN_TC=$SCRIPTS/recaser/train-truecaser.perl
TC=$SCRIPTS/recaser/truecase.perl

data_dir=$2
dest_dir=$3/data_deen
mkdir -p $dest_dir

sl=en
tl=de

file_name=train

# 1.Tokenizer
perl $TOKENIZER -a -l $sl < $data_dir/$file_name.$sl > $dest_dir/$file_name.tok.$sl
perl $TOKENIZER -a -l $tl < $data_dir/$file_name.$tl > $dest_dir/$file_name.tok.$tl

# 2. Truecaser 
# Train
perl $TRAIN_TC -corpus $dest_dir/$file_name.tok.$sl -model $dest_dir/truecase-model.$sl
perl $TRAIN_TC -corpus $dest_dir/$file_name.tok.$tl -model $dest_dir/truecase-model.$tl

# Apply
perl $TC -model $dest_dir/truecase-model.$sl < $dest_dir/$file_name.tok.$sl > $dest_dir/$file_name.tc.$sl
perl $TC -model $dest_dir/truecase-model.$tl < $dest_dir/$file_name.tok.$tl > $dest_dir/$file_name.tc.$tl

# 3. clean
perl $CLEAN  $dest_dir/$file_name.tc $sl $tl  $dest_dir/$file_name.clean.$sl 1 100

