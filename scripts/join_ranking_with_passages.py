import pandas as pd
import io
import gzip
from tqdm import tqdm

#rankings_path = '/u/scr/ashwinp/research/readi/experiments/colbert-rerank/15/ranking.tsv'
rankings_path = '/scr/biggest/ashwinp/experiments/colbert-rerank/19/ranking.tsv'
#output_path = '/u/scr/ashwinp/research/readi/experiments/colbert-rerank/15/ranking_passages.tsv'
output_path = '/scr/biggest/ashwinp/experiments/colbert-rerank/19/ranking_passages.tsv'
#passages_path = '/u/scr/ashwinp/research/readi/data/dpr-wiki/psgs_w100.tsv.gz'
passages_path = '/scr/biggest/ashwinp/readi/data/dpr-wiki/psgs_w100.tsv.gz'
rankings_df = pd.read_csv(rankings_path, sep='\t', names=['qid', 'pid', 'rank', 'score'])
joined_rankings_dfs = []

with io.TextIOWrapper(io.BufferedReader(gzip.open(passages_path))) as file:
    for chunk_df in tqdm(pd.read_csv(file, sep='\t', chunksize=1000000, names=['pid', 'text', 'title'], header=0)):
        chunk_df.set_index('pid', inplace=True)
        joined_rankings_dfs.append(rankings_df.join(chunk_df, on='pid', how='inner'))


joined_rankings_df = pd.concat(joined_rankings_dfs).sort_values(by=['qid', 'rank'])
joined_rankings_df.to_csv(output_path, sep='\t', index=False)



