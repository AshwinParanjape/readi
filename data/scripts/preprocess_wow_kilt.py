import json
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("input_file_path", type=str, help="path to kilt train.jsonl")
parser.add_argument("output_folder_path", type=str, help="path to output folder")
parser.add_argument("output_file_prefix", type=str, help="prefix of output files")
args = parser.parse_args()

sources = []
targets = []
missing_output = False
with open(args.input_file_path) as f:
    for line in f:
        instance = json.loads(line)
        sources.append(instance['input'].replace('\n', ' | '))
        if 'output' in instance:
            targets.append(instance['output'][0]['answer'])
        else:
            missing_output=True

with open(Path(args.output_folder_path)/Path(f"{args.output_file_prefix}.source"), 'w') as f:
    for s in sources: f.write(s+'\n')

if not missing_output:
    with open(Path(args.output_folder_path)/Path(f"{args.output_file_prefix}.target"), 'w') as f:
        for t in targets: f.write(t+'\n')
