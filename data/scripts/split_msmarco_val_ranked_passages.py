import pandas as pd
import argparse
from pathlib import Path
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument("input_file_path", type=str, help="path to msmarco ranked_passages.tsv")
split_point = 6233
args = parser.parse_args()

input_file_path = Path(args.input_file_path)
with open(input_file_path.parent/'new_val_ranking_passages.tsv', 'w') as f1o:
    with open(input_file_path.parent/'new_test_ranking_passages.tsv', 'w') as f2o:
        with open(args.input_file_path) as f:
            header = f.readline()
            f1o.write(header)
            f2o.write(header)
            for line in tqdm(f):
                qid = line.strip().split('\t')[0]
                if int(qid) < split_point:
                    f1o.write(line)
                else:
                    qid, rest = line.split('\t', 1)
                    qid = int(qid) - split_point
                    new_line = '\t'.join([str(qid), rest])
                    f2o.write(new_line)
