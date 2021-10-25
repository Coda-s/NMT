import argparse
from multiprocessing import Pool
import os

def write_file(list, file_name):
    with open(file_name, 'w', encoding='utf-8') as f:
        for i in list:
            f.write(i + '\n')

def match(filepatch, file_name, src_lang, tgt_lang, fdict_name, save_path):
    fsrc_name = os.path.join(filepatch, file_name + '.' + src_lang)
    ftgt_name = os.path.join(filepatch, file_name + '.' + tgt_lang)
    with open(fsrc_name, 'r', encoding='utf-8') as fsrc,\
        open(ftgt_name, 'r', encoding='utf-8') as ftgt,\
        open(fdict_name, 'r', encoding='utf-8') as  fdict:
        term_dict = list(set([term.strip() for term in fdict.readlines()]))
        src_term_list = [term.split('\t')[0] for term in term_dict]
        tgt_term_list = [term.split('\t')[1] for term in term_dict]
        matched_src_lines, matched_tgt_lines = [], []
        term_lines = []
        for src_line, tgt_line in zip(fsrc.readlines(), ftgt.readlines()):
            src_line, tgt_line = src_line.strip(), tgt_line.strip()
            term_line = []
            save_flag = False
            for src_term, tgt_term in zip(src_term_list, tgt_term_list):
                if len(term_line) >= 3:
                    break
                if src_term in src_line:
                    term_start_pos = src_line.index(src_term)
                    term_end_pos = term_start_pos + len(src_term)
                    if term_start_pos == 0:
                        new_src_term = src_term + ' '
                    elif term_end_pos == len(src_term):
                        new_src_term = ' ' + src_term
                    else:
                        new_src_term = ' ' + src_term + ' '
                    if new_src_term not in src_line:
                        continue
                    tgt_term = tgt_term.split(' &#124; ')
                    for tgt in tgt_term:
                        if tgt in tgt_line:
                            term_start_pos = tgt_line.index(tgt)
                            term_end_pos = term_start_pos + len(tgt)
                            if term_start_pos == 0:
                                new_tgt = tgt + ' '
                            elif term_end_pos == len(tgt):
                                new_tgt = ' ' + tgt
                            else:
                                new_tgt = ' ' + tgt + ' '
                            if new_tgt not in tgt_line:
                                continue
                            save_flag = True
                            term = src_term + ' ||| ' + tgt
                            flag = True
                            for word in term_line:
                                if term in word:
                                    flag = False
                            if term not in term_line and len(term_line) < 3 and flag == True:
                                term_line.append(term)
                            # break
            if save_flag == True:
                matched_src_lines.append(src_line)
                matched_tgt_lines.append(tgt_line)
                term_line.sort(key=lambda i: len(i), reverse=True)
                term_lines.append('\t'.join(term_line))
        fsrc_out_name = os.path.join(save_path, file_name + '.' + src_lang)
        ftgt_out_name = os.path.join(save_path, file_name + '.' + tgt_lang)
        ftermlines_out_name = os.path.join(save_path, file_name)
        write_file(matched_src_lines, fsrc_out_name + '.matched')
        write_file(matched_tgt_lines, ftgt_out_name + '.matched')
        write_file(term_lines, ftermlines_out_name + '.term_lines')

def run(args, i):
    file_name = args.split + str(i)
    match(args.file_path, file_name, args.s, args.t, args.fdict, args.save_path)

def creat_batch(fin_name, src_lang, tgt_lang, batch_num):
    fsrc_name = fin_name + '.tok.' + src_lang
    ftgt_name = fin_name + '.tok.' + tgt_lang
    with open(fsrc_name, 'r', encoding='utf-8') as fsrc,\
        open(ftgt_name, 'r', encoding='utf-8') as ftgt:   # + '.sample'
        fsrc_all_lines, ftgt_all_lines = fsrc.readlines(), ftgt.readlines()
        
        all_num = len(fsrc_all_lines)
        # all_num = 1019
        every_num = int(all_num / batch_num)
        for i in range(batch_num):
            start = every_num*i
            end = every_num * (i + 1)
            if i == batch_num - 1:
                end = all_num
            fsrc_batch_lines, ftgt_batch_lines = fsrc_all_lines[start:end], ftgt_all_lines[start:end]
            fsrcout = open(fin_name + str(i+1) + '.' + src_lang, 'w', encoding='utf-8')
            ftgtout = open(fin_name + str(i+1) + '.' + tgt_lang, 'w', encoding='utf-8')
            for fsrc_line, ftgt_line in zip(fsrc_batch_lines, ftgt_batch_lines):
                fsrcout.write(fsrc_line)
                ftgtout.write(ftgt_line)

def merge(file_name, lang = None, batch_num=10):
    if lang != None:
        with open(file_name + '.matched.' + lang, 'w', encoding='utf-8') as fout:
            for i in range(batch_num):
                with open(file_name + str(i+1) + '.' + lang + '.matched', 'r', encoding='utf-8') as fin:
                    for line in fin.readlines():
                        fout.write(line)
    else:
        with open(file_name + '.term_lines', 'w', encoding='utf-8') as fout:
            for i in range(batch_num):
                with open(file_name + str(i+1) + '.term_lines', 'r', encoding='utf-8') as fin:
                    for line in fin.readlines():
                        fout.write(line)

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--file_path', default='./dataset/')
    parse.add_argument('--split', default='test')
    parse.add_argument('--save_path', default='./dataset/matched')
    parse.add_argument('--s', default='en')
    parse.add_argument('--t', default='de')
    parse.add_argument('--fdict', default='./dict/en-de.filter')
    parse.add_argument('--batch_num', default=1, type=int)
    args = parse.parse_args()

    creat_batch(os.path.join(args.file_path, args.split), args.s, args.t, args.batch_num)

    p = Pool(args.batch_num)  # 创建含有batch_num个进程的进程池
    results = []  # 存放每一个进程返回的结果
    for i in range(args.batch_num):  # 启动batch_num个进程
        r = p.apply_async(run, args=(args, i+1,))  # 产生一个非同步进程，函数的参数用args传递
        results.append(r)  # 将返回结果放入results
    p.close()  # 关闭进程池
    p.join()  # 结束
    # for i in results:
    #     print(i.get())
    merge(os.path.join(args.save_path, args.split), args.s, args.batch_num)
    merge(os.path.join(args.save_path, args.split), args.t, args.batch_num)
    merge(os.path.join(args.save_path, args.split), batch_num=args.batch_num)