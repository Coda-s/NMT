work_dir=$1
dict_dir=$2
data_dir=$work_dir/phrase
table=$data_dir/phrase-table
table_derpt=$data_dir/phrase.table.derpt
table_clean=$data_dir/phrase.table.clean
vocab_filter=$data_dir/phrase.vocab
th_num=8

# Step 1:词表去重——去除重复句，处理一对多or多对多
line_num=$(cat $table | wc -l) 
echo $line_num
evety_num=`expr $line_num / $th_num`
split -d -$evety_num $table $table

python deal_table.py --deal_name table_de_repeated \
    --fin_name $table \
    --fout_name $table_derpt \
    --pool_num $th_num 

cat ${table_derpt}* > $table_derpt.merge


# Step 2:词表清洗——去除带标点的、首尾是停用词的、大于3-gram的短语，包含短语选择最长的
line_num=$(cat $table_derpt.merge | wc -l) 
echo $line_num
evety_num=`expr $line_num / $th_num`
split -d -$evety_num $table_derpt.merge $table_derpt

python deal_table.py --deal_name clean_table \
    --fin_name $table_derpt \
    --fout_name $table_clean \
    --pool_num $th_num 

cat $table_clean* > $table_clean.merge

# 删除切分成多份的数据

rm $data_dir/*0*
# rm $data_dir/*1*
# rm $data_dir/*2*
# rm $data_dir/*3*
# rm $data_dir/*4*
# rm $data_dir/*5*
# rm $data_dir/*6*
# rm $data_dir/*7*
# rm $data_dir/*8*

# Step 3:词表筛选——只保留得分在阈值内的数据
python deal_table.py --deal_name filter_vocab \
    --fin_name $table_clean.merge \
    --fout_name $vocab_filter

# Step 4:词表拷贝到词典路径
cp  $vocab_filter $dict_dir/en-de.txt