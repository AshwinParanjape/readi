import pandas as pd
import json
import csv

collection_path = '../CORDv1/corpus/collection.tsv'
#meta_df = pd.read_json(meta_path, orient='records')

collection_df = pd.read_csv(collection_path, sep='\t', quoting=csv.QUOTE_NONE, doublequote=False, lineterminator='\n', quotechar=None, usecols=['id', 'cid'])
collection_df = collection_df.rename(columns={'id': 'pid'})
collection_df = collection_df.set_index('pid')


for exp, split in [('118', 'train'), ('117', 'train'), ('116', 'dev'), ('115', 'dev')]:
    meta_path = f'../CORDv1/corpus/{split}.meta'
    with open(meta_path, 'r') as f:
        meta_df = pd.DataFrame([json.loads(l.strip()) for l in f.readlines()])
    meta_df.index.rename('qid')
    passages_df = pd.read_csv(f'/scr/biggest/ashwinp/experiments/colbert-rerank/{exp}/ranking.tsv', sep='\t', quoting=csv.QUOTE_NONE, doublequote=False, lineterminator='\n', quotechar=None, names=['qid', 'pid', 'rank', 'score'], header=None)
    passages_df = passages_df.set_index('pid').join(collection_df, how='left').reset_index()
    joined_passages_df = passages_df.set_index('qid').join(meta_df['cord_uid'], how='left')
    print(len(joined_passages_df))
    filtered_passages_df = joined_passages_df.query("cord_uid != cid").copy()
    print(len(filtered_passages_df))
    filtered_passages_df.index.name = 'qid'
    filtered_passages_df = filtered_passages_df.reset_index()
    filtered_passages_df = filtered_passages_df.sort_values(['qid', 'rank'])
    filtered_passages_df['rank'] = filtered_passages_df.groupby('qid')['score'].rank(method='first', ascending=False).astype('int32')
    truncated_filtered_passages_df = filtered_passages_df.query('rank<=100')
    truncated_filtered_passages_df.to_csv(f'/scr/biggest/ashwinp/experiments/colbert-rerank/{exp}/truncated_ranking.tsv', sep='\t', header=None, columns=['qid', 'pid', 'rank', 'score'])
