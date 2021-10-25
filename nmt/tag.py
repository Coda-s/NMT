import argparse
import os
import numpy as np

parse = argparse.ArgumentParser()
parse.add_argument('--match_path', required=True, type=str)
parse.add_argument('--output_path', required=True, type=str)
parse.add_argument('--slang', required=True, type=str)
parse.add_argument('--tlang', required=True, type=str)
parse.add_argument('--split', choices=["train", "valid", "test"], type=str)
args = parse.parse_args()

def is_in(A, B):
    return any([A == B[i:i+len(A)] for i in range(0, len(B)-len(A)+1)])

def write_file(list, file_name):
    with open(file_name, 'w', encoding='utf-8') as f:
        for i in list:
            f.write(i + '\n')

def replace_by_tag(split, slang, tlang, is_train=True):
    with open(os.path.join(args.match_path, split+'.matched.'+slang), 'r', encoding='utf-8') as fsrc, \
        open(os.path.join(args.match_path, split+'.matched.'+tlang), 'r', encoding='utf-8') as ftgt, \
        open(os.path.join(args.match_path, split+'.term_lines'), 'r', encoding='utf-8') as fterm_lines:
        src_lines = [line.strip() for line in fsrc.readlines()]
        tgt_lines = [line.strip() for line in ftgt.readlines()]
        term_lines = [line.strip() for line in fterm_lines.readlines()]

        idxs = [idx for idx in range(len(src_lines))]
        if is_train == True:
            choose_num = len(src_lines) // 10
            choose = np.random.choice(idxs, choose_num, replace=False)
        else:
            choose = idxs[:]

        src_unreplace, tgt_unreplace = list(), list()
        src_res = [list() for i in range(8)]
        tgt_res = [list() for i in range(8)]
        for idx, (src_line, tgt_line, term_line) in enumerate(zip(src_lines, tgt_lines, term_lines)):
            if idx not in choose:
                src_res[0].append(src_line)
                tgt_res[0].append(tgt_line)
            else:
                term_line = term_line.split('\t')
                term_line = [item.split(' ||| ') for item in term_line]
                # method 1
                # ---------------------------------------------------------------------------------
                src_line1 = src_line
                tgt_line1 = tgt_line
                label_count = 1
                for term_pair in term_line:
                    src_word = term_pair[0]
                    tgt_word = term_pair[1]
                    src_line1 = src_line1.replace(src_word, "<TAG{}>".format(label_count))
                    tgt_line1 = tgt_line1.replace(tgt_word, "<TAG{}>".format(label_count))
                    label_count += 1

                # method 2
                # ---------------------------------------------------------------------------------
                src_line2 = src_line
                tgt_line2 = tgt_line
                for term_pair in term_line:
                    src_word = term_pair[0]
                    tgt_word = term_pair[1]
                    src_line2 = src_line2.replace(src_word, tgt_word)

                # method 3
                # ---------------------------------------------------------------------------------
                src_line3 = src_line
                tgt_line3 = tgt_line
                for term_pair in term_line:
                    src_word = term_pair[0]
                    tgt_word = term_pair[1]
                    src_line3 = src_line3.replace(src_word, "<TAG> " + tgt_word + " <\TAG>")
                    tgt_line3 = tgt_line3.replace(tgt_word, "<TAG> " + tgt_word + " <\TAG>")

                # method 4
                # ---------------------------------------------------------------------------------
                src_line4 = src_line
                tgt_line4 = tgt_line
                for term_pair in term_line:
                    src_word = term_pair[0]
                    tgt_word = term_pair[1]
                    src_line4 = src_line4.replace(src_word, src_word + " <TAG> " + tgt_word + " <\TAG>")
                    tgt_line4 = tgt_line4.replace(tgt_word, "<TAG> " + tgt_word + " <\TAG>")

                # method 5
                # ---------------------------------------------------------------------------------
                src_line5 = src_line
                tgt_line5 = tgt_line
                for term_pair in term_line:
                    src_word = term_pair[0]
                    tgt_word = term_pair[1]
                    src_line5 = src_line5.replace(src_word, "<TAG> " + src_word + " <SEP> " + tgt_word + " <\TAG>")
                    
                # method 6
                # ---------------------------------------------------------------------------------
                src_line6 = src_line
                tgt_line6 = tgt_line
                for term_pair in term_line:
                    src_word = term_pair[0]
                    tgt_word = term_pair[1]
                    src_line6 += " <TAG> " + src_word + " <SEP> " + tgt_word + " <\TAG>"

                # method 7
                # ---------------------------------------------------------------------------------
                src_line7 = src_line
                tgt_line7 = tgt_line
                label_count = 1
                for term_pair in term_line:
                    src_word = term_pair[0]
                    tgt_word = term_pair[1]
                    src_line7 = src_line7.replace(src_word, "<TAG> " + src_word + " <\TAG>")
                    src_line7 += " <TAG> " + tgt_word + " <\TAG>"
                    label_count += 1
                # ---------------------------------------------------------------------------------

                src_res[1].append(src_line1)
                src_res[2].append(src_line2)
                src_res[3].append(src_line3)
                src_res[4].append(src_line4)
                src_res[5].append(src_line5)
                src_res[6].append(src_line6)
                src_res[7].append(src_line7)

                tgt_res[1].append(tgt_line1)
                tgt_res[2].append(tgt_line2)
                tgt_res[3].append(tgt_line3)
                tgt_res[4].append(tgt_line4)
                tgt_res[5].append(tgt_line5)
                tgt_res[6].append(tgt_line6)
                tgt_res[7].append(tgt_line7)

        if is_train == False:
            for i in range(1, 8):
                src_res[i].extend(src_res[0])
                tgt_res[i].extend(tgt_res[0])
        for i in range(1, 7):
            assert len(src_res[i]) == len(src_res[7])
        

        for i in range(1, 8):
            target_dir = os.path.join(args.output_path, "labeled_{}".format(i))
            os.system("mkdir -p {}".format(target_dir))
            with open(os.path.join(target_dir, split+'.label.'+slang), "w", encoding="UTF-8") as f:
                for line in src_res[i]:
                    f.write(line + '\n')
            with open(os.path.join(target_dir, split+'.label.'+tlang), "w", encoding="UTF-8") as f:
                for line in tgt_res[i]:
                    f.write(line + '\n')


def test_replace_by_tag(split, slang, tlang,):
    with open(os.path.join(args.match_path, split+'.matched.'+slang), 'r', encoding='utf-8') as fsrc,\
        open(os.path.join(args.match_path, split+'.matched.'+tlang) , 'r', encoding='utf-8') as ftgt,\
        open(os.path.join(args.match_path, split+'.term_lines'), 'r', encoding='utf-8') as fterm_lins:
        
        src_lines = [line.strip() for line in fsrc.readlines()]
        tgt_lines = [line.strip() for line in ftgt.readlines()]
        term_lines = [line.strip() for line in fterm_lins.readlines()]

        src_res = [list() for i in range(8)]
        # tgt_res = [list() for i in range(8)]
        tag_save = list()
        for idx, (src_line, term_line) in enumerate(zip(src_lines, term_lines)):

            term_line = term_line.split('\t')
            term_line = [item.split(' ||| ') for item in term_line]
            line_tag = []
            # method 1
            # ---------------------------------------------------------------------------------
            src_line1 = src_line
            # tgt_line1 = tgt_line
            label_count = 1

            for term_pair in term_line:
                src_word = term_pair[0]
                tgt_word = term_pair[1]
                src_line1 = src_line1.replace(src_word, "<TAG{}>".format(label_count))
                line_tag.append("<TAG{}>".format(label_count) + ' ||| ' + tgt_word)
                # tgt_line1 = tgt_line1.replace(tgt_word, "<TAG{}>".format(label_count))
                label_count += 1

            # method 2
            # ---------------------------------------------------------------------------------
            src_line2 = src_line
            for term_pair in term_line:
                src_word = term_pair[0]
                tgt_word = term_pair[1]
                src_line2 = src_line2.replace(src_word, tgt_word)

            # method 3
            # ---------------------------------------------------------------------------------
            src_line3 = src_line
            for term_pair in term_line:
                src_word = term_pair[0]
                tgt_word = term_pair[1]
                src_line3 = src_line3.replace(src_word, "<TAG> " + tgt_word + " <\TAG>")

            # method 4
            # ---------------------------------------------------------------------------------
            src_line4 = src_line
            for term_pair in term_line:
                src_word = term_pair[0]
                tgt_word = term_pair[1]
                src_line4 = src_line4.replace(src_word, src_word + " <TAG> " + tgt_word + " <\TAG>")
            # method 5
            # ---------------------------------------------------------------------------------
            src_line5 = src_line
            for term_pair in term_line:
                src_word = term_pair[0]
                tgt_word = term_pair[1]
                src_line5 = src_line5.replace(src_word, "<TAG> " + src_word + " <SEP> " + tgt_word + " <\TAG>")

            # method 6
            # ---------------------------------------------------------------------------------
            src_line6 = src_line
            for term_pair in term_line:
                src_word = term_pair[0]
                tgt_word = term_pair[1]
                src_line6 += " <TAG> " + src_word + " <SEP> " + tgt_word + " <\TAG>"

            # method 7
            # ---------------------------------------------------------------------------------
            src_line7 = src_line
            label_count = 1
            for term_pair in term_line:
                src_word = term_pair[0]
                tgt_word = term_pair[1]
                src_line7 = src_line7.replace(src_word,
                                              "<TAG> " + src_word + " <\TAG>")
                src_line7 += " <TAG> " + tgt_word + " <\TAG>"
                label_count += 1
            # ---------------------------------------------------------------------------------

            src_res[1].append(src_line1)
            src_res[2].append(src_line2)
            src_res[3].append(src_line3)
            src_res[4].append(src_line4)
            src_res[5].append(src_line5)
            src_res[6].append(src_line6)
            src_res[7].append(src_line7)

            tag_save.append('\t'.join(line_tag))


        for i in range(1, 8):
            src_res[i].extend(src_res[0])
        for i in range(1, 7):
            assert len(src_res[i]) == len(src_res[7])
        

        for i in range(1, 8):
            target_dir = os.path.join(args.output_path, "labeled_{}".format(i))
            os.system("mkdir -p {}".format(target_dir))
            with open(os.path.join(target_dir, split+'.label.'+slang), "w", encoding="UTF-8") as f:
                for line in src_res[i]:
                    f.write(line + '\n')
            with open(os.path.join(target_dir, split+'.label.'+tlang), "w", encoding="UTF-8") as f:
                for line in tgt_lines:
                    f.write(line + '\n')

if __name__ == '__main__':

    if args.split == "test":
        test_replace_by_tag(args.split, args.slang, args.tlang)
    elif args.split == "train":
        replace_by_tag(args.split, args.slang, args.tlang)
    elif args.split == "valid":
        replace_by_tag(args.split, args.slang, args.tlang, is_train=False)