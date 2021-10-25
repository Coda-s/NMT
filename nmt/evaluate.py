import os
import sacrebleu
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, type=str)
parser.add_argument('--clean', action="store_true")
args = parser.parse_args()

def get_sentences(path, type):
    path = os.path.join(path, "test" + type)
    with open(path, "r", encoding="UTF-8") as f:
        lines = [line.strip() for line in f.readlines()]
    return lines
        

def get_vocab(path):
    with open(path, "r", encoding="UTF-8") as f:
        lines = [line.strip() for line in f.readlines()]
    vocab = list()
    for line in lines:
        line = line.split('\t')
        line_res = [line[i].split(" ||| ")[1] for i in range(len(line))]
        vocab.append(line_res)
    return vocab

def cal_tur(predict_sentence, golden_sentence, vocab):
    true_num, total_num = 0, 0
    assert len(predict_sentence) == len(golden_sentence)
    assert len(predict_sentence) == len(vocab)
    for predict, golden, words in zip(predict_sentence, golden_sentence, vocab):
        total_num += len(words)
        for word in words:
            if word in predict:
                true_num += 1
    return true_num, total_num

if __name__ == "__main__":
    
    
    data_path = os.path.join("result", args.model)
    vocab_path = "data/test.term_lines"
    
    vocab = get_vocab(vocab_path)
    
    if args.clean:
        prediction_sentences = get_sentences(data_path, ".clean.prediction")
    else:
        prediction_sentences = get_sentences(data_path, ".prediction")
    golden_sentences = get_sentences(data_path, ".tgt")
    bleu = sacrebleu.corpus_bleu(prediction_sentences, [golden_sentences], force=True)
    true_num, total_num = cal_tur(prediction_sentences, golden_sentences, vocab)
    print("bleu = {:.2f} | term_use_rate = {:.2f}% ({:4d} / {:4d})".format \
        (bleu.score, 100.0*true_num/total_num, true_num, total_num))