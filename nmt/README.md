### **Lexical constraint translation using data augmentation**


#### **Requirements**
```python
python>=3.6
torch=1.8.0
sacrebleu=1.5.0
```

#### **Installation**
```shell
cd fairseq
pip install --editable ./
```

#### **Usage**
##### **preparation**
```shell
data
   |-test.raw.en
   |-test.raw.de
   |-test.term_lines
   |-train.raw.en
   |-train.raw.de
   |-valid.raw.en
   |-valid.raw.de
   |-vocab.txt

------------------------------------------------------

vocab.txt:
Source word and target word in a line and are saperated by '\t' 

eg.
interview	Interview
prize	Preisgeld

------------------------------------------------------

test.term_lines
Word pair which both occur in source and target
All pairs are saperated by '\t' and two words in a pair are seperated by ' ||| '

eg.
Syria ||| Syrien
consensus ||| Konsens	Syria ||| Syrien
sponsorship ||| Sponsoring	sport ||| Sport

------------------------------------------------------

```

##### **preprocess**
```shell
bash preprocess.sh
```

##### **train**
```shell
bash train.sh $name
```

##### **predict**
```shell
bash predict.sh $name
```

#### **Using Leca**
```shell
1. add special token "<seq>" into dictionary
fairseq/fairseq/data/dictionary.py

in function __init__()
add argument: sep="<sep>"
add init method: self.sep_index = self.add_symbol(sep)

in class Dictionary()
add function: 
def sep(self):
   return self.sep_index

2. train
bash train_leca.sh
```