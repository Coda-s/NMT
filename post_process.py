import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default="baseline", help='input file')
args = parser.parse_args()

with open(args.name + "/test.prediction", "r", encoding="UTF-8") as f:
    lines = [line.strip() for line in f.readlines()]

res = []
for line in lines:
    line = line.replace("<TAG>", "")
    line = line.replace("<\\TAG>", "")
    line = re.sub(r"[\s]+", " ", line)
    res.append(line)

with open(args.name + "/test.clean.prediction", "w", encoding="UTF-8") as f:
    for line in res:
        f.write(line + '\n')


