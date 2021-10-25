import argparse
from multiprocessing import Pool
from nltk.corpus import stopwords
from tqdm import tqdm

def table_de_repeated(fin_name, fout_name, i):
    name = str(0) + str(i) if i < 10 else str(i)
    with open(fin_name + name, 'r', encoding='utf-8') as fin,\
        open(fout_name + name, 'w', encoding='utf-8') as fout:
        src_list, tgt_list = [], []
        for i, line in enumerate(fin.readlines()):
            if i % 1000 == 0:
                print(i)
            line = line.strip().split(' ||| ')
            src, tgt = line[0], line[1]
            if src not in src_list and tgt not in tgt_list:
                src_list.append(src)
                tgt_list.append(tgt)
                fout.write(' ||| '.join(line) + '\n')

def clean_table(fin_name, fout_name, i):
    name = str(0) + str(i) if i < 10 else str(i)
    with open(fin_name + name, 'r', encoding='utf-8') as fin, \
            open(fout_name + name, 'w', encoding='utf-8') as fout:
        src_list, tgt_list, new_lines = [], [], []
        list_id = 0
        zh_puncs = ['，', '。', '？', '！', '；', '、', '：', '；', '‘', '’', '“', '”', '—', '——']
        en_puncs = [',', '.', '?', '!', ';', ':', ';',  '"', '-', '#', '/', '–', '−', '%', '$', '�',
			 '●', '•', '·', '≤', '(', ')', '€', '…', '@', '[', ']', '˚', '„', '‒', '‑', ]
        en_stops = stopwords.words('english')
        puns = list(set(zh_puncs + en_puncs))
        for line in tqdm(fin.readlines()):
            line = line.strip().split(' ||| ')
            src, tgt = line[0], line[1]
            if any([l in src for l in puns]) or any([l in tgt for l in puns]):
                continue
            if src.split(' ')[-1] in en_stops or src.split(' ')[0] in en_stops:
                continue
            if len(src.split(' ')) < 1 or len(src.split(' ')) > 3:
                continue
            if list_id > 0:
                if (src_list[list_id - 1] in src) or (tgt_list[list_id - 1] in tgt):
                    src_list[list_id - 1] = src
                    tgt_list[list_id - 1] = tgt
                    new_lines[list_id - 1] = ' ||| '.join(line)
                else:
                    tgt_list.append(tgt)
                    src_list.append(src)
                    new_lines.append(' ||| '.join(line))
                    list_id += 1
            else:
                tgt_list.append(tgt)
                src_list.append(src)
                new_lines.append(' ||| '.join(line))
                list_id += 1
        for line in new_lines:
            fout.write(line + '\n')

def filter_vocab(fin_name, fout_name):
    with open(fin_name, 'r', encoding='utf-8') as fin, \
            open(fout_name, 'w', encoding='utf-8') as fout:
        for line in tqdm(fin.readlines()):
            line = line.strip().split(' ||| ')
            src, tgt = line[0], line[1]
            score = line[2].split()
            score = [float(s) for s in score]
            if score[0] > 0.5 and score[2] > 0.5 and score[1] > 0.03 and score[3] > 0.03:
                fout.write(src + '\t' + tgt + '\n')



if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--deal_name', default='')
    parse.add_argument('--fin_name', default='')
    parse.add_argument('--fout_name', default='')
    parse.add_argument('--pool_num', default=1, type=int)
    parse.add_argument('--fsrc_name', default='')
    parse.add_argument('--ftgt_name', default='')
    parse.add_argument('--fvocab_name', default='')
    args = parse.parse_args()
    if args.deal_name == 'table_de_repeated':
        p = Pool(args.pool_num)  
        results = []  
        for i in range(args.pool_num):  
            r = p.apply_async(table_de_repeated, args=(args.fin_name, args.fout_name, i, ))
            results.append(r)  
        p.close() 
        p.join() 
        for i in results:
            print(i.get())
    elif args.deal_name == 'clean_table':
        p = Pool(args.pool_num+1) 
        results = []  
        for i in range(args.pool_num): 
            r = p.apply_async(clean_table, args=(args.fin_name, args.fout_name, i,))
            results.append(r)  
        p.close()  
        p.join() 
        for i in results:
            print(i.get())
    elif args.deal_name == 'filter_vocab':
        filter_vocab(args.fin_name, args.fout_name)
    else:
        print('deal error !!!')