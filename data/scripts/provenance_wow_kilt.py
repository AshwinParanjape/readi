import json
import argparse
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import csv

parser = argparse.ArgumentParser()
parser.add_argument("input_file_path", type=str, help="path to kilt train.jsonl")
parser.add_argument("input_ks_path", type=str, help="path to chunked kilt knowledgesource tsv: kilt_ks_chunks.tsv")
parser.add_argument("output_folder_path", type=str, help="path to output folder")
parser.add_argument("output_file_prefix", type=str, help="prefix of output files")
args = parser.parse_args()

provenances = []
missing_output = False

# Collect provenances in a list
with open(args.input_file_path) as f:
    for line in f:
        instance = json.loads(line)
        if 'provenance' in instance['output'][0]:
            provenance = instance['output'][0]['provenance'][0]
            provenances.append(provenance)
        else:
            provenances.append(None)
example_instante = {'id': '6bc20426-99d6-11ea-8a20-773209e30a7b_0',
 'input': 'I like to watch ice hockey on TV. My favorite team is the Chicago Blackhawks.',
 'output': [{'answer': "The Blackhawks are one of my favorite teams, they've won 6 Stanley Cup Championships since they started in 1926",
   'provenance': [{'wikipedia_id': '73126',
     'title': 'Chicago Blackhawks',
     'start_paragraph_id': 1,
     'start_character': 260,
     'end_paragraph_id': 1,
     'end_character': 333,
     'bleu_score': 1.0,
     'section': 'Section::::Abstract.'}]}]}
# Create a mapping from wiki id to paragraphs
doc_2_prov_mapping = defaultdict(list)
for prov_id, p in enumerate(provenances):
    if p is not None:
        doc_2_prov_mapping[p['wikipedia_id']].append((prov_id, (p['start_paragraph_id']-1, p['start_character']), (p['end_paragraph_id']-1, p['end_character']), p['bleu_score']))

doc_exists = {d: False for d in doc_2_prov_mapping}



with open(args.input_ks_path, 'r', newline='') as csvfile:
    ks_reader = csv.reader(csvfile, delimiter='\t', quoting=csv.QUOTE_NONE, doublequote=False, lineterminator='\n', quotechar=None)
    for row in tqdm(ks_reader):
        if row[0] == 'id':
            continue
        pid, text, title, wid = row[0:4]
        if wid in doc_2_prov_mapping:
            doc_exists[wid] = True
            start_pid, start_char, end_pid, end_char = row[6:10]
            passage_start = (int(start_pid), int(start_char))
            passage_end = (int(end_pid), int(end_char))
            for prov_id, prov_start, prov_end, bleu_score in doc_2_prov_mapping[wid]:
                if prov_end < passage_start or prov_start > passage_end:
                    continue
                if 'pid' in provenances[prov_id]:
                    print("Repeated provenance", provenances[prov_id], pid, title, text)
                provenances[prov_id]['pid']=pid
                provenances[prov_id]['title']=title
                provenances[prov_id]['text']=text
                provenances[prov_id]['bleu_score']=bleu_score

print("Out of the provenances:")
print("Documents missing for ", sum([len(doc_2_prov_mapping[d]) for d, e in doc_exists.items()]))
print("Documents exist but no matching provenance for", len([d for d, prov in doc_2_prov_mapping.items() if 'pid' not in prov and doc_exists[d]]))


with open(Path(args.output_folder_path)/Path(f"{args.output_file_prefix}.provenance"), 'w') as f:
    outfilewriter = csv.writer(f, delimiter='\t', quoting=csv.QUOTE_NONE, doublequote=False, lineterminator='\n', quotechar=None)
    outfilewriter.writerow(['qid','pid','rank','bleu_score','text','title'])
    for qid, prov in enumerate(provenances):
        if prov is None:
            outfilewriter.writerow((qid, -1))
        elif 'pid' in prov:
            outfilewriter.writerow((qid, prov['pid'], 0, prov['bleu_score'], prov['text'], prov['title']))
        else:
            outfilewriter.writerow((qid, -1))


