import pandas as pd
import io
import gzip
from tqdm import tqdm
import csv

##rankings_path = '/u/scr/ashwinp/research/readi/experiments/colbert-rerank/15/ranking.tsv'
#rankings_path = '/scr/biggest/ashwinp/experiments/colbert-rerank/73/ranking.tsv'
##output_path = '/u/scr/ashwinp/research/readi/experiments/colbert-rerank/15/ranking_passages.tsv'
#output_path = '/scr/biggest/ashwinp/experiments/colbert-rerank/73/ranking_passages.tsv'
##passages_path = '/u/scr/ashwinp/research/readi/data/dpr-wiki/psgs_w100.tsv.gz'
##passages_path = '/scr/biggest/ashwinp/readi/data/dpr-wiki/psgs_w100.tsv.gz'
#passages_path = '/scr/biggest/ashwinp/readi/data/kilt/kilt_ks_chunks.tsv'

for expid in ['115', '116', '117', '118']:
    #rankings_path = f'/scr/biggest/ashwinp/experiments/colbert-rerank/{expid}/ranking.tsv'
    rankings_path = f'/scr/biggest/ashwinp/experiments/colbert-rerank/{expid}/truncated_ranking.tsv'
    output_path = f'/scr/biggest/ashwinp/experiments/colbert-rerank/{expid}/ranking_passages.tsv'
    passages_path = '/scr/biggest/ashwinp/readi/data/CORDv1/corpus/collection.tsv'
    rankings_df = pd.read_csv(rankings_path, sep='\t', names=['qid', 'pid', 'rank', 'score'])
    joined_rankings_dfs = []


    if passages_path[-2:] == 'gz':
        fileh = io.TextIOWrapper(io.BufferedReader(gzip.open(passages_path)))
    else:
        fileh = open(passages_path)

    with fileh:
        #for chunk_df in tqdm(pd.read_csv(fileh, sep='\t', chunksize=1000000, names=['pid', 'text', 'title'], header=0, usecols=['pid', 'text', 'title'], quoting=csv.QUOTE_NONE, doublequote=False, lineterminator='\n', quotechar=None)):
        for chunk_df in tqdm(pd.read_csv(fileh, sep='\t', chunksize=1000000, names=['pid', 'text', 'title', 'cid'], header=0, usecols=['pid', 'text', 'title', 'cid'], quoting=csv.QUOTE_NONE, doublequote=False, lineterminator='\n', quotechar=None)):
            chunk_df.set_index('pid', inplace=True)
            joined_rankings_dfs.append(rankings_df.join(chunk_df, on='pid', how='inner'))


    joined_rankings_df = pd.concat(joined_rankings_dfs).sort_values(by=['qid', 'rank'])
    joined_rankings_df.to_csv(output_path, sep='\t', quoting=csv.QUOTE_NONE, doublequote=False, line_terminator='\n', quotechar=None, index=False)
#joined_rankings_df.to_csv(output_path, sep='\t', index=False)



