import re
import argparse


parse = argparse.ArgumentParser()
parse.add_argument('--input_file', required=True, type=str)
parse.add_argument('--output_file', required=True, type=str)
args = parse.parse_args()

def repl(match):
    return match.group().replace("@@ ", "")

if __name__ == "__main__":
    with open(args.input_file, "r", encoding="UTF-8") as fin, \
        open(args.output_file, "w", encoding="UTF-8") as fout:
        lines = [line.strip() for line in fin.readlines()]
        for line in lines:
            line = re.sub(r"<.*>", repl, line)
            fout.write(line + '\n') 