import gzip
import io
import json
import types
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Optional

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.profiler import AdvancedProfiler
from torch.utils.data import DataLoader
from transformers import BartForConditionalGeneration, BartTokenizer, BertPreTrainedModel, BertModel, BertTokenizerFast, \
    BatchEncoding
from transformers.file_utils import ModelOutput
from transformers.modeling_outputs import Seq2SeqLMOutput
import string
import sys, os
import torch
import pandas as pd
from typing import List, Dict, Union
from torch.utils.data._utils.collate import default_collate
import argparse
from pathlib import Path
from meticulous import Experiment
from tqdm import tqdm

print(os.getcwd())
sys.path = ['retriever/ColBERT'] + sys.path
from colbert.modeling.tokenization import QueryTokenizer, DocTokenizer
from colbert.modeling.tokenization.utils import _sort_by_length, _split_into_batches
from colbert.utils.utils import load_checkpoint

node_rank = int(os.environ.get("NODE_RANK", 0))
local_rank = int(os.environ.get("LOCAL_RANK", 0))

# Special tokens
DOC_TOKEN = '[DOC]' # Separates input and doc when fed as context to the generator
TEXT_TOKEN = '|' # Separates title and text

from pytorch_lightning.loggers.base import LightningLoggerBase, rank_zero_only

def truncate_sequences_from_beginning_helper(
    self,
    ids: List[int],
    pair_ids: Optional[List[int]] = None,
    num_tokens_to_remove: int = 0,
    truncation_strategy = "longest_first",
    stride: int = 0,
) -> Tuple[List[int], List[int], List[int]]:
    """ Only truncates single sequences (doesn't work for pairs) and truncates from the beginning.
    Ignores pair_ids, truncation_strategy"""

    if num_tokens_to_remove <= 0:
        return ids, pair_ids, []

    overflowing_tokens = []
    for _ in range(num_tokens_to_remove):
        if not overflowing_tokens:
            window_len = min(len(ids), stride + 1)
        else:
            window_len = 1
        overflowing_tokens.extend(ids[:window_len])
        ids = ids[1:]

    return (ids, pair_ids, overflowing_tokens)

class MeticulousLogger(LightningLoggerBase):
    def __init__(self, experiment):
        super().__init__()
        self.exp = experiment

    @property
    def version(self):
        return self.curexpdir

    def log_hyperparams(self):
        pass

    @rank_zero_only
    def log_hyperparams(self, params):
        # params is an argparse.Namespace
        # your code to record hyperparameters goes here
        with self.exp.open('pl_params.json', 'w') as f:
            json.dump(f, vars(params))


    @rank_zero_only
    def log_metrics(self, metrics, step):
        # metrics is a dictionary of metric names and values
        # your code to record metrics goes here
        with self.exp.open('pl_metrics.json', 'w') as f:
            json.dump(f, vars(params))
        pass

    def save(self):
        # Optional. Any code necessary to save logger data goes here
        pass

    @rank_zero_only
    def finalize(self, status):
        # Optional. Any code that needs to be run after training
        # finishes goes here
        pass

def stable_softmax(input, dim, *args, **kwargs):
    c = input.max(dim=dim, keepdim=True).values
    return torch.nn.functional.softmax(input - c, dim)


class ColBERT(BertPreTrainedModel):
    def __init__(self, config, query_maxlen, doc_maxlen, mask_punctuation=True, dim=128, similarity_metric='cosine'):
        super(ColBERT, self).__init__(config)

        self.query_maxlen = query_maxlen
        self.doc_maxlen = doc_maxlen
        self.similarity_metric = similarity_metric
        self.dim = dim

        self.mask_punctuation = mask_punctuation
        self.skiplist = {}

        if self.mask_punctuation:
            self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
            self.skiplist = {w: True
                             for symbol in string.punctuation
                             for w in [symbol, self.tokenizer.encode(symbol, add_special_tokens=False)[0]]}

        self.bert = BertModel(config)
        self.linear = torch.nn.Linear(config.hidden_size, dim, bias=False)

        self.init_weights()

    def forward(self, Q, D):
        #q_input_ids, q_attention_masks = *Q
        #d_input_ids, d_attention_masks = *D
        #Q, D = combined_query_doc(q_input_ids, q_attention_masks, d_input_ids, d_attention_masks)
        #return self.score(Q, D)
        return self.score(self.query(*Q), self.doc(*D))

    def combined_query_doc(self, q_input_ids, q_attention_masks, d_input_ids, d_attention_masks):
        input_ids = torch.cat([q_input_ids, d_input_ids], dim=0)
        attention_masks = torch.cat([q_attention_masks, d_attention_masks])
        bert_embeddings = self.bert(input_ids, attention_mask=attention_masks)[0]
        embeddings = self.linear(bert_embeddings)
        Q = embeddings[:q_input_ids.shape[0], ...]
        D = embeddings[q_input_ids.shape[0]:, ...]
        mask = torch.tensor(self.mask(input_ids)).pin_memory().to(D, non_blocking=True).unsqueeze(2)
        D = D * mask
        Q = torch.nn.functional.normalize(Q, p=2, dim=2)
        D = torch.nn.functional.normalize(D, p=2, dim=2)
        return Q, D

    def query(self, input_ids, attention_mask):
        #input_ids, attention_mask = input_ids.to(DEVICE), attention_mask.to(DEVICE)
        Q = self.bert(input_ids, attention_mask=attention_mask)[0]
        Q = self.linear(Q)

        return torch.nn.functional.normalize(Q, p=2, dim=2)

    def doc(self, input_ids, attention_mask, keep_dims=True):
        #input_ids, attention_mask = input_ids.to(DEVICE), attention_mask.to(DEVICE)
        D = self.bert(input_ids, attention_mask=attention_mask)[0]
        D = self.linear(D)

        #mask = torch.tensor(self.mask(input_ids), device=DEVICE).unsqueeze(2).float()
        mask = torch.tensor(self.mask(input_ids)).pin_memory().to(D, non_blocking=True).unsqueeze(2)
        D = D * mask

        D = torch.nn.functional.normalize(D, p=2, dim=2)

        if not keep_dims:
            D, mask = D.cpu().to(dtype=torch.float16), mask.cpu().bool().squeeze(-1)
            D = [d[mask[idx]] for idx, d in enumerate(D)]

        return D
    def score(self, Q, D):
        if len(D.shape) == 3: # D has shape N x ld x emb
            if self.similarity_metric == 'cosine':
                return (Q @ D.permute(0, 2, 1)).max(2).values.sum(1)

            assert self.similarity_metric == 'l2'
            return (-1.0 * ((Q.unsqueeze(2) - D.unsqueeze(1))**2).sum(-1)).max(-1).values.sum(-1)

        #TODO: step through the code to make sure it is correct
        if len(D.shape) == 4: # D has shape N x nd x ld x emb
            assert len(Q.shape)==3, f"Q.shape = {Q.shape}; should be N x lq x emb"
            if self.similarity_metric == 'cosine':
                Q = Q.unsqueeze(1)
                D = D.permute(0, 1, 3, 2)
                return (Q @ D).max(3).values.sum(2)

            assert self.similarity_metric == 'l2'
            return (-1.0 * ((Q.unsqueeze(2).unsqueeze(1) - D.unsqueeze(2))**2).sum(-1)).max(-1).values.sum(-1)
            # returns a tensor of shape N x nd

    def mask(self, input_ids):
        mask = [[(x not in self.skiplist) and (x != 0) for x in d] for d in input_ids.cpu().tolist()]
        return mask

class ClosedSetRetrievals():
    """Iterable Dataset that loads the closed set of retrieved passages in chunks and also works with multiple workers"""
    def __init__(self, path: str):
        self.path = Path(path)

    def __iter__(self):
        if self.path.suffix=='.gz':
            opener = lambda path: io.TextIOWrapper(io.BufferedReader(gzip.open(self.path)))
        else:
            opener = lambda path: open(path, 'r')
        with opener(self.path) as file:
            last_qid = None
            last_qid_retrievals = None
            for chunk_df in pd.read_csv(file, sep='\t', chunksize=100000,
                                        names=['qid', 'pid', 'rank', 'score', 'doc_text', 'title'], header=0,
                                        dtype={'qid': int, 'pid': int, 'rank': int, 'score': float, 'doc_text':str,
                                               'title':str} ,
                                        na_filter=False):
                for qid, retrievals in chunk_df.groupby('qid'):
                    retrievals['text'] = retrievals['title'].str.cat(retrievals['doc_text'], sep=f' {TEXT_TOKEN} ')

                    # The last qid, which will had an incomplete set of retrievals, will be completed here
                    if last_qid == qid:
                        last_qid_retrievals = pd.concat([last_qid_retrievals, retrievals])

                    # The last qid has completed
                    else:
                        #Yield it and set the current one as the last qid
                        if last_qid_retrievals is not None:
                            yield last_qid, last_qid_retrievals
                        last_qid_retrievals = retrievals
                        last_qid=qid

            else: # For the last chunk return yield the last qid
                yield last_qid, last_qid_retrievals


class ColBERTScorer(ColBERT):
    def __init__(self, *args, truncate_query_from_start=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.query_tokenizer = QueryTokenizer(self.query_maxlen, truncate_from_start=truncate_query_from_start)
        self.doc_tokenizer = DocTokenizer(self.doc_maxlen)

    def tensorize(self, queries, batched_docs):
        docs = self.doc_tokenizer.tensorize([str(doc) for docs in batched_docs for doc in docs])
        queries = self.query_tokenizer.tensorize(queries)
        return queries, docs

    def forward(self, queries, batched_docs):
        n_instances = len(batched_docs)
        n_docs = len(batched_docs[0])
        Q, D = self.tensorize(queries, batched_docs)
        Q = [t.pin_memory().to(device=self.device, non_blocking=True) for t in Q]
        D = [t.pin_memory().to(device=self.device, non_blocking=True) for t in D]
        #query_embeds, doc_embeds = self.combined_query_doc(*Q, *D)
        query_embeds = self.query(*Q)
        doc_embeds = self.doc(*D)
        doc_embeds = doc_embeds.view(n_instances, n_docs, *doc_embeds.shape[1:])
        score = self.score(query_embeds, doc_embeds)
        return score
class DocumentSampler:
    def __call__(self, retrievals):
        raise NotImplementedError("Sampler needs to implement __call__method")

class RandomDocumentSampler(DocumentSampler):
    def __init__(self, n):
        self.n = n

    def __call__(self, retrievals: pd.DataFrame):
        # retrievals has columns ['qid', 'pid', 'rank', 'score', 'doc_text', 'title', 'text']
        if len(retrievals) == 0:
            return retrievals
        if self.n > len(retrievals):
            print("Fewer retrievals than n", sys.stderr)
            n = len(retrievals)
        else:
            n= self.n

        return retrievals.sample(n=n)

class TopKDocumentSampler(DocumentSampler):
    def __init__(self, k):
        self.k = k

    def __call__(self, retrievals: pd.DataFrame):
        # retrievals has columns ['qid', 'pid', 'rank', 'score', 'doc_text', 'title', 'text']
        if len(retrievals) == 0:
            return retrievals
        if self.k > len(retrievals):
            print("Fewer retrievals than k", sys.stderr)
            k = len(retrievals)
        else:
            k= self.k

        top_k_retrievals = retrievals.sort_values('score', ascending=False)[:k]
        return top_k_retrievals

class SimpleDocumentSampler(DocumentSampler):
    def __init__(self, n, temperature=1, top_k=None):
        self.n = n
        self.temperature=temperature
        assert n<=top_k, f"top_k={top_k} should at least be n={n}"
        self.top_k = top_k

    def __call__(self, retrievals: pd.DataFrame):
        # retrievals has columns ['qid', 'pid', 'rank', 'score', 'doc_text', 'title', 'text']
        if len(retrievals) == 0:
            return retrievals
        if self.n > len(retrievals):
            print("Fewer retrievals than n", sys.stderr)
            n = len(retrievals)
        else:
            n= self.n

        top_k = self.top_k or len(retrievals)
        retrievals['probs'] = stable_softmax(torch.tensor(retrievals['score'].values) / self.temperature, dim=0)
        top_k_retrievals = retrievals.sort_values('probs', ascending=False)[:top_k]
        return top_k_retrievals.sample(n=n, weights='probs')

class GuidedNoIntersectionDocumentSampler(DocumentSampler):
    def __init__(self, n, temperature=1, top_k=None):
        self.n = n
        self.temperature=temperature
        assert n<=top_k, f"top_k={top_k} should at least be n={n}"
        self.top_k = top_k
        self.p_minus_q_sampler = SimpleDocumentSampler(self.n//2, self.temperature, self.top_k)
        self.q_minus_p_sampler = SimpleDocumentSampler(self.n//2, self.temperature, self.top_k)

    def __call__(self, retrievals: pd.DataFrame):
        # retrievals has columns ['qid', 'pid', 'p_score', 'score_q', 'doc_text', 'title', 'text']
        p_minus_q = retrievals[(retrievals['score_p'].notna()) & (retrievals['score_q'].isna())].copy()
        p_minus_q['score'] = p_minus_q['score_p']
        p_minus_q_samples = self.p_minus_q_sampler(p_minus_q)

        q_minus_p = retrievals[(retrievals['score_q'].notna()) & (retrievals['score_p'].isna())].copy()
        q_minus_p['score'] = q_minus_p['score_q']
        q_minus_p_samples = self.q_minus_p_sampler(q_minus_p)

        mixed_samples = pd.concat([p_minus_q_samples, q_minus_p_samples])

        # If not enough samples were gotten (because some of the three sets not containing enough to sample from)
        # Add a few more on an ad-hoc basis
        if len(mixed_samples) < self.n:
            diff = self.n - len(mixed_samples)
            q_docs = retrievals[(retrievals['score_q'].notna())].copy()
            q_docs['score'] = q_docs['score_q']
            extra_samples = SimpleDocumentSampler(diff, self.temperature, self.top_k)(q_docs)
            mixed_samples = pd.concat([mixed_samples, extra_samples])

        return mixed_samples

class RankPNDocumentSampler(DocumentSampler):
    def __init__(self, n, kP=20, kQR=5, kQ1=20, kQ2=50, positives_cutoff=3):
        assert kQ1 <= kQ2, "Positives should be a subset of protected"
        self.n = n
        self.kP = kP
        self.kQR = kQR
        self.kQ1 = kQ1
        self.kQ2 = kQ2
        self.positives_cutoff = max(positives_cutoff, self.n//4+1) #at least one more than the number of positive samples desired
        self.positives_sampler = RandomDocumentSampler(self.n//4)
        self.negatives_sampler = RandomDocumentSampler(self.n//2-1)
        self.relevant_positive_sampler = RandomDocumentSampler(self.n//4)
        self.random_sampler = RandomDocumentSampler(1)

    def __call__(self, retrievals: pd.DataFrame, unrelated_retrievals: pd.DataFrame=None):
        protected_indices = (retrievals['rank_p'] <= self.kP) | (retrievals['rank_q'] <= self.kQ1)
        positives = retrievals[(retrievals['rank_p'] <= self.kP) & (retrievals['rank_q'] <= self.kQ2)]
        #if len(positives) == 0:
        #    return None

        positives = positives.sort_values('rank_p')[:self.positives_cutoff]
        positive_samples = self.positives_sampler(positives)

        relevant_positives = retrievals[(retrievals['rank_q'] <= self.kQR) & ~(retrievals['pid'].isin(positive_samples['pid']))]
        relevant_positive_samples = self.relevant_positive_sampler(relevant_positives)

        negatives = retrievals[~(protected_indices) & retrievals['rank_p'].notna()]
        negative_samples = self.negatives_sampler(negatives)

        #empty_sample = pd.DataFrame([{
            #'qid': retrievals['qid'][0], 'pid': -1, 'score_p': -1, 'score_q': -1,
            #'doc_text': '',  'title': '', 'text': ''
        #}])

        if unrelated_retrievals is not None:
            unrelated_samples = self.random_sampler(unrelated_retrievals)
            mixed_samples = pd.concat([relevant_positive_samples, positive_samples, negative_samples, unrelated_samples])
        else:
            mixed_samples = pd.concat([relevant_positive_samples, positive_samples, negative_samples])
        diff = self.n - len(mixed_samples)
        if diff > 0:
            extra_samples = RandomDocumentSampler(diff)(negatives)
            if unrelated_retrievals is not None:
                mixed_samples = pd.concat([relevant_positive_samples, positive_samples, negative_samples, extra_samples, unrelated_samples])
            else:
                mixed_samples = pd.concat([relevant_positive_samples, positive_samples, negative_samples, extra_samples])

        return mixed_samples

class GuidedDocumentSampler(DocumentSampler):
    def __init__(self, n, temperature=1, top_k=None):
        self.n = n
        self.temperature=temperature
        assert n<=top_k, f"top_k={top_k} should at least be n={n}"
        self.top_k = top_k
        self.p_intersection_q_sampler = SimpleDocumentSampler(self.n//4, self.temperature, self.top_k)
        self.p_minus_q_sampler = SimpleDocumentSampler(self.n//4, self.temperature, self.top_k)
        self.q_minus_p_sampler = SimpleDocumentSampler(self.n//4, self.temperature, self.top_k)
        self.random_sampler = RandomDocumentSampler(self.n//4)

    def __call__(self, retrievals: pd.DataFrame, unrelated_retrievals: pd.DataFrame=None):
        # retrievals has columns ['qid', 'pid', 'score_p', 'score_q', 'doc_text', 'title', 'text']
        p_intersection_q = retrievals[(retrievals['score_p'].notna()) & (retrievals['score_q'].notna())].copy()
        p_intersection_q['score'] = p_intersection_q['score_q']
        p_intersection_q_samples = self.p_intersection_q_sampler(p_intersection_q)

        p_minus_q = retrievals[(retrievals['score_p'].notna()) & (retrievals['score_q'].isna())].copy()
        p_minus_q['score'] = p_minus_q['score_p']
        p_minus_q_samples = self.p_minus_q_sampler(p_minus_q)

        q_minus_p = retrievals[(retrievals['score_q'].notna()) & (retrievals['score_p'].isna())].copy()
        q_minus_p['score'] = q_minus_p['score_q']
        q_minus_p_samples = self.q_minus_p_sampler(q_minus_p)

        if unrelated_retrievals is not None:
            unrelated_samples = self.random_sampler(unrelated_retrievals)
            mixed_samples = pd.concat([p_intersection_q_samples, p_minus_q_samples, q_minus_p_samples, unrelated_samples])
        else:
            mixed_samples = pd.concat([p_intersection_q_samples, p_minus_q_samples, q_minus_p_samples])


        # If not enough samples were gotten (because some of the three sets not containing enough to sample from)
        # Add a few more on an ad-hoc basis
        if len(mixed_samples) < self.n:
            diff = self.n - len(mixed_samples)
            q_docs = retrievals[(retrievals['score_q'].notna()) & ~(retrievals['pid'].isin(mixed_samples['pid']))].copy()
            q_docs['score'] = q_docs['score_q']
            extra_samples = SimpleDocumentSampler(diff, self.temperature, self.top_k)(q_docs)
            if unrelated_retrievals is not None:
                mixed_samples = pd.concat([p_intersection_q_samples, p_minus_q_samples, q_minus_p_samples, extra_samples, unrelated_samples])
            else:
                mixed_samples = pd.concat([p_intersection_q_samples, p_minus_q_samples, q_minus_p_samples, extra_samples])

        return mixed_samples

class PosteriorDocumentSampler(DocumentSampler):
    def __init__(self, n, temperature=1, top_k=None):
        self.n = n
        self.temperature=temperature
        assert n<=top_k, f"top_k={top_k} should at least be n={n}"
        self.top_k = top_k
        self.p_intersection_q_sampler = SimpleDocumentSampler(self.n//2, self.temperature, self.top_k)
        self.q_minus_p_sampler = SimpleDocumentSampler(self.n//2, self.temperature, self.top_k)

    def __call__(self, retrievals: pd.DataFrame, unrelated_retrievals: pd.DataFrame=None):
        # retrievals has columns ['qid', 'pid', 'score_p', 'score_q', 'doc_text', 'title', 'text']
        p_intersection_q = retrievals[(retrievals['score_p'].notna()) & (retrievals['score_q'].notna())].copy()
        p_intersection_q['score'] = p_intersection_q['score_q']
        p_intersection_q_samples = self.p_intersection_q_sampler(p_intersection_q)

        q_minus_p = retrievals[(retrievals['score_q'].notna()) & (retrievals['score_p'].isna())].copy()
        q_minus_p['score'] = q_minus_p['score_q']
        q_minus_p_samples = self.q_minus_p_sampler(q_minus_p)

        mixed_samples = pd.concat([p_intersection_q_samples, q_minus_p_samples])


        # If not enough samples were gotten (because some of the three sets not containing enough to sample from)
        # Add a few more on an ad-hoc basis
        if len(mixed_samples) < self.n:
            diff = self.n - len(mixed_samples)
            q_docs = retrievals[(retrievals['score_q'].notna()) & ~(retrievals['pid'].isin(mixed_samples['pid']))].copy()
            q_docs['score'] = q_docs['score_q']
            extra_samples = SimpleDocumentSampler(diff, self.temperature, self.top_k)(q_docs)
            mixed_samples = pd.concat([p_intersection_q_samples, q_minus_p_samples, extra_samples])

        return mixed_samples

class Seq2SeqDataset(torch.utils.data.IterableDataset):
    def __init__(self, source_path: str, target_path: str, worker_id=0, n_workers=1):
        self.source = pd.read_csv(source_path, sep='\t', names=['source'], dtype=str, na_filter=False)
        self.target = pd.read_csv(target_path, sep='\t', names=['target'], dtype=str, na_filter=False)
        self.worker_id = worker_id
        self.n_workers = n_workers

    def __iter__(self):
        for qid, (source, target) in enumerate(zip(self.source['source'], self.target['target'])):
            if qid % self.n_workers == self.worker_id and qid < len(self)*self.n_workers:  # This query belongs to this worker
                yield {'qid': qid,
                       'source': source,
                       'target': target,
                       }

    def __len__(self):
        return len(self.source)//self.n_workers

def recompute_retriever_scores(scorer: ColBERTScorer, query: str, retrievals_df: pd.DataFrame):
    queries = [query]
    batched_docs = [retrievals_df['text'].tolist()]
    scores = scorer(queries, batched_docs)
    rescored_retrievals_df = retrievals_df.copy()
    rescored_retrievals_df['score'] = scores[0, :].cpu()
    return rescored_retrievals_df

class PDataset(torch.utils.data.IterableDataset):
    def __init__(self, source_path: str, target_path: str, p_retrievals_path: str, sampler:DocumentSampler, worker_id=0, n_workers=1, p_scorer: ColBERTScorer = None, yield_scores=False):
        self.source = pd.read_csv(source_path, sep='\t', names=['source'], dtype=str, na_filter=False)
        if target_path:
            self.target = pd.read_csv(target_path, sep='\t', names=['target'], dtype=str, na_filter=False)
        else:
            self.target = pd.DataFrame()
            self.target['target'] = self.source['source'] # To quickly get the same shape
            self.target['target'] = ''
        self.p_retrievals = ClosedSetRetrievals(p_retrievals_path)
        self.p_scorer = p_scorer
        self.cached_scores: Dict[int, Dict[int, float]] = defaultdict(dict)
        self.sampler = sampler
        self.worker_id = worker_id
        self.n_workers = n_workers
        self.yield_scores = yield_scores

    def __iter__(self):
        for qid, (source, target, (p_qid, p_retrievals)) in enumerate(zip(self.source['source'], self.target['target'], self.p_retrievals)):
            #assert (qid == p_qid) , (qid, p_qid)
            if qid % self.n_workers == self.worker_id and qid < len(self)*self.n_workers:  # This query belongs to this worker
                if self.p_scorer:
                    p_retrievals = recompute_retriever_scores(self.p_scorer, source, p_retrievals)
                sampled_retrievals = self.sampler(p_retrievals)
                yield_dict= {'qid': qid,
                        'source': source,
                        'target': target,
                        'doc_ids': sampled_retrievals['pid'].tolist(),
                        'doc_texts': sampled_retrievals['text'].tolist(),
                        }
                if self.yield_scores:
                    yield_dict['doc_scores'] = torch.tensor(sampled_retrievals['score'].tolist())
                yield yield_dict
    def __len__(self):
        return len(self.source)//self.n_workers


class PQDataset(torch.utils.data.IterableDataset):
    def __init__(self, source_path:str, target_path: str, p_retrievals_path: str, q_retrievals_path: str, sampler: DocumentSampler, worker_id=0,n_workers=1, yield_scores=False, include_unrelated=True):
        self.source = pd.read_csv(source_path, sep='\t', names=['source'], dtype=str, na_filter=False)
        self.target = pd.read_csv(target_path, sep='\t', names=['target'], dtype=str, na_filter=False)
        self.p_retrievals = ClosedSetRetrievals(p_retrievals_path)
        self.q_retrievals = ClosedSetRetrievals(q_retrievals_path)
        #p_samples_list = []
        #q_samples_list = []
        #for qid, (source, target, (p_qid, p_retrievals), (q_qid, q_retrievals)) in tqdm(enumerate(zip(self.source['source'], self.target['target'], self.p_retrievals, self.q_retrievals))):
        #    p_samples_list.append(p_retrievals.sample(n=1))
        #    q_samples_list.append(q_retrievals.sample(n=1))

        #p_samples = pd.concat(p_samples_list)
        #q_samples = pd.concat(q_samples_list)
        #self.unrelated_retrievals = p_samples.merge(q_samples, how='outer', on=['qid', 'pid', 'doc_text', 'title', 'text'], suffixes = ('_p', '_q'))
        self.unrelated_retrievals = None
        self.sampler = sampler
        self.worker_id = worker_id
        self.n_workers = n_workers
        self.skipped_instances = 0
        self.yield_scores = yield_scores
        self.include_unrelated = include_unrelated

    def __iter__(self):
        # Important: doc_scores are the Q retriever scores
        for qid, (source, target, (p_qid, p_retrievals), (q_qid, q_retrievals)) in enumerate(zip(self.source['source'], self.target['target'], self.p_retrievals, self.q_retrievals)):
            #assert (qid == p_qid) and (qid == q_qid), (qid, p_qid, q_qid)
            if qid % self.n_workers == self.worker_id:  # This query belongs to this worker
                merged_retrievals = p_retrievals.merge(q_retrievals, how='outer', on=['qid', 'pid', 'doc_text', 'title', 'text'], suffixes = ('_p', '_q'))
                sampled_retrievals = self.sampler(merged_retrievals, self.unrelated_retrievals)
                #sampled_retrievals = self.sampler(merged_retrievals)
                if sampled_retrievals is None:
                    self.skipped_instances+=1
                    continue
                sampled_retrievals['score_q'] = sampled_retrievals['score_q'].fillna(merged_retrievals['score_q'].min())
                yield_dict = {'qid': qid,
                        'source': source,
                        'target': target,
                        'doc_ids': sampled_retrievals['pid'].tolist(),
                        'doc_texts': sampled_retrievals['text'].tolist()
                }

                if self.yield_scores:
                    yield_dict['doc_scores'] = torch.tensor(sampled_retrievals['score_q'].tolist())

                yield yield_dict
                if self.include_unrelated:
                    if self.unrelated_retrievals is not None:
                        self.unrelated_retrievals = pd.concat([self.unrelated_retrievals, merged_retrievals.sample(n=10)])
                        if len(self.unrelated_retrievals)>2000:
                            self.unrelated_retrievals.sample(2000)
                    else:
                        self.unrelated_retrievals = merged_retrievals.sample(n=2)

    #def __len__(self):
    #    return (len(self.source)//self.n_workers)-self.skipped_instances


# TODO: override collate function to simply collate tuples into a list
def collate_fn(batch: Dict):
    collated = default_collate(
        [{k:v for k, v in d.items() if k in {'qid', 'source', 'target', 'doc_scores'}} for d in batch]
    )
    collated['doc_ids'] = [d['doc_ids'] for d in batch ]
    collated['doc_texts'] = [d['doc_texts'] for d in batch ]
    if 'doc_scores' in batch[0]:
        collated.update(default_collate([{'doc_scores': d['doc_scores']} for d in batch]))
    return collated

@dataclass
class GeneratorOutput(Seq2SeqLMOutput):
    input_encoding: Dict = None
    output_encoding: Dict = None

class Generator(torch.nn.Module):
    def __init__(self, generator, tokenizer):
        super().__init__()
        self.generator = generator
        self.tokenizer = tokenizer

    def prepare_generator_inputs(self, sources, batched_docs=None):
        if batched_docs:
            generator_inputs = [f'{source} {DOC_TOKEN} {doc}' for source, docs in zip(sources, batched_docs) for doc in
                                docs]
        else:
            generator_inputs = [f'{source}' for source in sources]

        input_encoding: BatchEncoding = self.tokenizer(generator_inputs, padding=True, return_tensors='pt', truncation=True,
                                                       max_length=256, pad_to_multiple_of=8)
        input_encoding.data = {n: t.pin_memory().to(device=self.generator.device, non_blocking=True) for n, t in
                               input_encoding.data.items()}
        return input_encoding

    def prepare_training_inputs(self, sources, targets, batched_docs=None):
        if batched_docs:
            generator_inputs = [f'{source} {DOC_TOKEN} {doc}' for source, docs in zip(sources, batched_docs) for doc in docs]
            generator_outputs = [f'{target}' for target, docs in zip(targets, batched_docs) for doc in docs]
        else:
            generator_inputs = [f'{source}' for source in sources]
            generator_outputs = [f'{target}' for target in targets]
        input_encoding: BatchEncoding = self.tokenizer(generator_inputs, padding=True, return_tensors='pt', truncation=True, max_length=256, pad_to_multiple_of=8)
        input_encoding.data = {n: t.pin_memory().to(device=self.generator.device, non_blocking=True) for n, t in input_encoding.data.items()}


        output_encoding: BatchEncoding = self.tokenizer(generator_outputs, padding=True, return_tensors='pt', truncation=True, max_length=64, pad_to_multiple_of=8)
        output_encoding.data = {n: t.pin_memory().to(device=self.generator.device, non_blocking=True) for n, t in output_encoding.data.items()}
        return input_encoding, output_encoding


    def get_target_logits(self, input_encoding, output_encoding):
        lm_output : Seq2SeqLMOutput = self.generator(
            input_ids = input_encoding['input_ids'],
            attention_mask = input_encoding['attention_mask'],
            decoder_input_ids = output_encoding['input_ids'],
            return_dict=True
        )
        return lm_output

    def forward(self, sources, targets, batched_docs=None):
        input_encoding, output_encoding = self.prepare_training_inputs(sources, targets, batched_docs)
        lm_output = self.get_target_logits(input_encoding, output_encoding)
        lm_output.input_encoding = input_encoding
        lm_output.output_encoding = output_encoding
        return lm_output


    def generate(self, sources, batched_docs=None, **generation_kwargs)->ModelOutput:
        input_encoding = self.prepare_generator_inputs(sources, batched_docs)
        generator_output = self.generator.generate(input_ids=input_encoding['input_ids'],
                                        attention_mask=input_encoding['attention_mask'],
                                         return_dict_in_generate=True,
                                        **generation_kwargs)
        # Use the following to generate based on pure sampling and verify that the same probabilities are computed by
        # the rescorer as well
        #generator_output = self.generator.generate(input_ids=input_encoding['input_ids'],
        #                                                    attention_mask=input_encoding['attention_mask'],
        #                                                    return_dict_in_generate=True, output_scores=True, do_sample=True, num_beams=1,
        #                                                    no_repeat_ngram_size=0,
        #                                                    min_length=-1,
        #                                                    forced_eos_token_id=None,
        #                                                    top_k=0,
        #                                                    **generation_kwargs)
        generator_output.log_liklihood = self.rescore_from_tensors(input_encoding, generator_output, generation_kwargs.get('num_return_sequences', 1))
        decoded_output = self.tokenizer.batch_decode(generator_output.sequences, skip_special_tokens=True)
        generator_output.strings = decoded_output
        generator_output.input_encoding = input_encoding
        return generator_output

    def rescore_from_tensors(self, input_encoding, generator_output, n_samples_per_doc):
        """
        This function computes the raw probabilities as assigned by the generator (tested with BartForConditionalGeneration)
        This is useful because the scores returned by generate are affected by warping and beam search, also useful
        for rescoring with another model that wasn't use to generate the sequences
        """
        rescorer_output = self.generator(input_ids=input_encoding['input_ids'].repeat_interleave(repeats=n_samples_per_doc, dim=0),
                                         attention_mask=input_encoding['attention_mask'].repeat_interleave(repeats=n_samples_per_doc, dim=0),
                                         decoder_input_ids=generator_output.sequences[:, :-1])
        rescorer_log_softmax=torch.log_softmax(rescorer_output.logits, dim=2)
        not_pad = generator_output.sequences[:, 1:-1]!=self.generator.config.pad_token_id
        rescorer_ll = not_pad * rescorer_log_softmax.gather(dim=2, index=generator_output.sequences[:, 1:-1].unsqueeze(2)).squeeze(2)
        rescorer_seq_ll = rescorer_ll.sum(dim=1)

        # To verify if the rescorer probs are correct, do pure sampling in the generator (uncomment lines in generate
        # function) and uncomment the following lines to get generator scores in the same form as rescorer_seq_ll
        #generator_log_softmax=torch.log_softmax(torch.stack(generator_output.scores).permute(1,0,2), dim=2)
        #generator_ll = not_pad * generator_log_softmax[:, :-1, :].gather(dim=2, index=generator_output.sequences[:, 1:-1].unsqueeze(2)).squeeze(2)
        #generator_seq_ll = generator_ll.sum(dim=1)
        return rescorer_seq_ll


class LM_NLL(torch.nn.Module):
    def __init__(self, generator: Generator):
        super().__init__()
        self.generator = generator

    def forward(self, sources: List[str], targets: List[str], batched_docs: List[List[str]]=None):
        lm_output = self.generator(sources, targets, batched_docs)
        labels = lm_output.output_encoding['input_ids'][:, 1:] #Shape: Bsz*nd x (len-1)
        labels = labels.permute(1, 0) #Shape: (len-1) x Bsz*nd
        logits = lm_output.logits[:, :-1, :] #Shape: Bsz*nd x (len-1) x vocab_size
        logits = logits.permute(1,2,0) #Shape: (len-1) x vocab_size x Bsz*nd
        nll = torch.nn.functional.cross_entropy(logits, labels, reduction='none').sum(dim=0) # Shape: Bsz*nd
        if batched_docs:
            bsz = len(batched_docs)
            nd = len(batched_docs[0])
            nll = nll.view(bsz, nd)
        return nll

@dataclass
class MarginalizedNLL():
    loss: Union[float, torch.Tensor]
    lm_nll: torch.Tensor
    p_scores: torch.Tensor
    def metrics(self):
        return [('loss', self.loss)]

    @property
    def intermediate_values(self):
        return [('lm_nll', self.lm_nll), ('p_scores', self.p_scores)]

class MarginalizedNLLFn(torch.nn.Module):
    def __init__(self, scorer:ColBERTScorer, generator: Generator):
        super().__init__()
        self.scorer = scorer
        self.generator = generator
        self.generator_nll = LM_NLL(self.generator)

    def forward(self, sources: List[str], targets: List[str], batched_docs: List[List[str]]):
        """
        :return: Marginalized NLL loss
        """
        scores = self.scorer(sources, batched_docs)
        doc_log_probs = torch.nn.functional.log_softmax(scores, dim=1) #Shape: n_instances x n_docs
        generator_log_prob = -self.generator_nll(sources, targets, batched_docs) #Shape: n_instances x n_docs
        loss = -torch.logsumexp(doc_log_probs + generator_log_prob, dim=1).sum(dim=0)
        return MarginalizedNLL(loss, lm_nll = -generator_log_prob, p_scores=scores)

@dataclass
class ELBO():
    loss: Union[float, torch.Tensor]
    reconstruction_score: Union[float, torch.Tensor]
    kl_divergence: Union[float, torch.Tensor]
    marginalized_loss: Union[float, torch.Tensor]
    lm_nll: torch.Tensor
    p_scores: torch.Tensor
    q_scores: torch.Tensor

    @property
    def metrics(self):
        return [('loss', self.loss),
                ('marginalized_loss', self.marginalized_loss),
                ('reconstruction_score', self.reconstruction_score),
                ('kl_divergence', self.kl_divergence)]

    @property
    def intermediate_values(self):
        return [('p_scores.tsv', self.p_scores),
                ('q_scores.tsv', self.q_scores),
                ('nll.tsv', self.lm_nll)]

class ELBOFn(torch.nn.Module):
    def __init__(self, p_scorer:ColBERTScorer, q_scorer: ColBERTScorer, generator: Generator):
        super().__init__()
        self.p_scorer = p_scorer
        self.q_scorer = q_scorer
        self.generator = generator
        self.generator_nll = LM_NLL(self.generator)

    def forward(self, sources: List[str], targets: List[str], batched_docs: List[List[str]]):
        st_text = [s + ' | ' +t for s, t in zip(sources, targets)]
        p_scores = self.p_scorer(sources, batched_docs)
        p_probs = stable_softmax(p_scores, dim=1)
        p_log_probs = torch.nn.functional.log_softmax(p_scores, dim=1) #Shape: n_instances x n_docs
        q_scores = self.q_scorer(st_text, batched_docs)
        q_probs = stable_softmax(q_scores, dim=1)
        generator_log_prob = -self.generator_nll(sources, targets, batched_docs) #Shape: n_instances x n_docs

        marginalized_nll_loss = -torch.logsumexp(p_log_probs + generator_log_prob, dim=1).sum(dim=0)

        reconstruction_score = (q_probs * generator_log_prob).sum()
        kl_regularization = (q_probs * (q_probs.log() - p_probs.log())).sum()
        elbo_loss = -(reconstruction_score - kl_regularization)

        return ELBO(elbo_loss, reconstruction_score, kl_regularization, marginalized_nll_loss, -generator_log_prob, p_scores, q_scores)


def log_value(filename, stage, epoch, batch_idx, key, value):
    epoch_filename = filename.parent / (filename.stem+'_'+str(epoch)+filename.suffix)
    with open(epoch_filename, 'a') as f:
        f.write(f'{stage}\t{epoch}\t{batch_idx}\t{key}\t{value}\n')

def log_batch_value(filename, stage, epoch, qids, batched_doc_ids, batched_values):
    epoch_filename = filename.parent / (filename.stem+'_'+str(epoch)+filename.suffix)
    with open(epoch_filename, 'a') as f:
        for qid, doc_ids, values in zip(qids, batched_doc_ids, batched_values):
            for doc_id, value in zip(doc_ids, values):
                f.write(f'{stage}\t{epoch}\t{qid}\t{doc_id}\t{value}\n')

def log_all(expdir, loop, current_epoch, batch_idx, batch, output):
    for name, value in output.metrics:
        log_value(Path(expdir) / Path('metrics.tsv'), loop, current_epoch, batch_idx, name, value)

    for fname, values in output.intermediate_values:
        log_batch_value(Path(expdir) / fname, loop, current_epoch, batch['qid'], batch['doc_ids'], values)

class InheritableCheckpointMixin():
    @classmethod
    def init_from_checkpoints(cls, state_dict, **init_kwargs):
        obj = cls(**init_kwargs)
        obj.load_state_dict(state_dict, strict=False)
        obj.set_loss_fn()
        return obj


class NLLLossSystem(pl.LightningModule):
    def __init__(self, expdir='', lr=1e-3, truncate_query_from_start=False) :
        super().__init__()
        self._generator = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
        self._generator_tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        #self.generator = Generator(self._generator, self._generator_tokenizer, truncate_from_start=truncate_query_from_start)
        self.generator = Generator(self._generator, self._generator_tokenizer)
        self.loss_fn = LM_NLL(self.generator)
        self.expdir=expdir
        self.lr = lr

    def training_step(self, batch, batch_idx):
        # ['qid': List[int], 'source':List[str], 'target':List[str], 'doc_ids': List[List[int]], 'doc_texts': List[List[str]]]
        output = self.loss_fn(batch['source'], batch['target'])

        log_value(Path(self.expdir)/ Path('metrics.tsv'), 'train', self.current_epoch, batch_idx, 'loss', output.sum())

        return output.sum()

    def validation_step(self, batch, batch_idx):
        # ['qid': List[int], 'source':List[str], 'target':List[str], 'doc_ids': List[List[int]], 'doc_texts': List[List[str]]]
        output = self.loss_fn(batch['source'], batch['target'])

        log_value(Path(self.expdir)/ Path('metrics.tsv'), 'valid', self.current_epoch, batch_idx, 'loss', output.sum())

        return output.sum()


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def on_train_epoch_start(self):
        self.setup_tsv_files()

    def setup_tsv_files(self):
        with open(Path(self.expdir) / Path(f'metrics_{self.current_epoch}.tsv'), 'w') as f:
            f.write('stage\tepoch\tbatch_idx\tkey\tvalue\n')

class MarginalizedLossSystem(pl.LightningModule, InheritableCheckpointMixin):
    def __init__(self, query_maxlen, doc_maxlen, expdir='', lr=1e-3, truncate_query_from_start=False) :
        super().__init__()
        self._generator = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
        self._generator_tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        #self._generator_tokenizer.add_tokens([DOC_TOKEN, TEXT_TOKEN])
        #self.generator = Generator(self._generator, self._generator_tokenizer, truncate_from_start=truncate_query_from_start)
        self.generator = Generator(self._generator, self._generator_tokenizer)
        self.p_scorer = ColBERTScorer.from_pretrained('bert-base-uncased',
                                          truncate_query_from_start = truncate_query_from_start,
                                          query_maxlen=query_maxlen,
                                          doc_maxlen=doc_maxlen,
                                          )
        #saved_state_dict = torch.load(p_scorer_checkpoint, map_location='cpu')
        #self.p_scorer.load_state_dict(saved_state_dict['model_state_dict'], strict=False)
        self.set_loss_fn()

        self.expdir = expdir
        self.lr = lr

    @staticmethod
    def extract_state_dict_from_colbert_checkpoints(p_scorer_checkpoint):
        p_scorer_checkpoint = torch.load(p_scorer_checkpoint, torch.device('cpu'))
        state_dict = {'p_scorer.'+k: v for k, v in p_scorer_checkpoint.items()}
        return state_dict

    def set_loss_fn(self):
        self.loss_fn = MarginalizedNLLFn(self.p_scorer, self.generator)

    @staticmethod
    def extract_state_dict_from_checkpoints(p_scorer_checkpoint, generator_checkpoint):
        p_scorer_checkpoint = torch.load(p_scorer_checkpoint, torch.device('cpu'))
        generator_checkpoint = torch.load(generator_checkpoint, torch.device('cpu'))
        state_dict = filter_state_dict(generator_checkpoint, 'generator')
        state_dict.update(filter_state_dict(generator_checkpoint, '_generator'))
        state_dict.update(filter_state_dict(p_scorer_checkpoint, 'p_scorer'))
        return state_dict

    def training_step(self, batch, batch_idx):
        # ['qid': List[int], 'source':List[str], 'target':List[str], 'doc_ids': List[List[int]], 'doc_texts': List[List[str]]]
        output: MarginalizedNLL = self.loss_fn(batch['source'], batch['target'], batch['doc_texts'])

        log_value(Path(self.expdir)/ Path('metrics.tsv'), 'train', self.current_epoch, batch_idx, 'loss', output.loss)

        log_batch_value(Path(self.expdir)/ Path('p_scores.tsv'), 'train', self.current_epoch, batch['qid'], batch['doc_ids'], output.p_scores)
        log_batch_value(Path(self.expdir)/ Path('nll.tsv'), 'train', self.current_epoch, batch['qid'], batch['doc_ids'], output.lm_nll)

        return output.loss

    def validation_step(self, batch, batch_idx):
        # ['qid': List[int], 'source':List[str], 'target':List[str], 'doc_ids': List[List[int]], 'doc_texts': List[List[str]]]
        output: MarginalizedNLL = self.loss_fn(batch['source'], batch['target'], batch['doc_texts'])

        log_value(Path(self.expdir)/ Path('metrics.tsv'), 'val', self.current_epoch, batch_idx, 'loss', output.loss)

        log_batch_value(Path(self.expdir)/ Path('p_scores.tsv'), 'val', self.current_epoch, batch['qid'], batch['doc_ids'], output.p_scores)
        log_batch_value(Path(self.expdir)/ Path('nll.tsv'), 'val', self.current_epoch, batch['qid'], batch['doc_ids'], output.lm_nll)
        return output.loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def on_train_epoch_start(self):
        self.setup_tsv_files()

    def setup_tsv_files(self):
        with open(Path(self.expdir) / Path(f'p_scores_{self.current_epoch}.tsv'), 'w') as f:
            f.write('stage\tepoch\tq_id\tdoc_id\tp_score\n')
        with open(Path(self.expdir) / Path(f'nll_{self.current_epoch}.tsv'), 'w') as f:
            f.write('stage\tepoch\tq_id\tdoc_id\tnll\n')
        with open(Path(self.expdir) / Path(f'metrics_{self.current_epoch}.tsv'), 'w') as f:
            f.write('stage\tepoch\tbatch_idx\tkey\tvalue\n')

class ELBOLossSystem(pl.LightningModule, InheritableCheckpointMixin):
    def __init__(self, query_maxlen, doc_maxlen, expdir='', lr=1e-3, truncate_query_from_start=False, p_scorer_checkpoint=None, q_scorer_checkpoint=None):
        super().__init__()
        self._generator = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
        self._generator_tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        #self._generator_tokenizer.add_tokens([DOC_TOKEN, TEXT_TOKEN])
        #self.generator = Generator(self._generator, self._generator_tokenizer, truncate_from_start=truncate_query_from_start)
        self.generator = Generator(self._generator, self._generator_tokenizer)
        self.p_scorer = ColBERTScorer.from_pretrained('bert-base-uncased',
                                          truncate_query_from_start = truncate_query_from_start,
                                          query_maxlen=query_maxlen,
                                          doc_maxlen=doc_maxlen)
        if p_scorer_checkpoint:
            saved_state_dict = torch.load(p_scorer_checkpoint, map_location='cpu')
            self.p_scorer.load_state_dict(saved_state_dict['model_state_dict'], strict=False)

        self.q_scorer = ColBERTScorer.from_pretrained('bert-base-uncased',
                                          truncate_query_from_start = truncate_query_from_start,
                                          query_maxlen=query_maxlen,
                                          doc_maxlen=doc_maxlen)
        if q_scorer_checkpoint:
            saved_state_dict = torch.load(q_scorer_checkpoint, map_location='cpu')
            self.q_scorer.load_state_dict(saved_state_dict['model_state_dict'], strict=False)
        self.set_loss_fn()
        self.lr = lr
        self.expdir = expdir


    def on_train_epoch_start(self):
        self.setup_tsv_files()

    @staticmethod
    def extract_state_dict_from_colbert_checkpoints(p_scorer_checkpoint, q_scorer_checkpoint):
        p_scorer_checkpoint = torch.load(p_scorer_checkpoint, torch.device('cpu'))
        q_scorer_checkpoint = torch.load(q_scorer_checkpoint, torch.device('cpu'))
        state_dict = {'p_scorer.'+k: v for k, v in p_scorer_checkpoint.items()}
        state_dict.update({'q_scorer.'+k: v for k, v in q_scorer_checkpoint.items()})
        return state_dict

    def set_loss_fn(self):
        self.loss_fn = ELBOFn(self.p_scorer, self.q_scorer, self.generator)

    @staticmethod
    def extract_state_dict_from_checkpoints(p_scorer_checkpoint, q_scorer_checkpoint, generator_checkpoint):
        p_scorer_checkpoint = torch.load(p_scorer_checkpoint, torch.device('cpu'))
        q_scorer_checkpoint = torch.load(q_scorer_checkpoint, torch.device('cpu'))
        generator_checkpoint = torch.load(generator_checkpoint, torch.device('cpu'))
        state_dict = filter_state_dict(generator_checkpoint, 'generator')
        state_dict.update(filter_state_dict(generator_checkpoint, '_generator'))
        state_dict.update(filter_state_dict(p_scorer_checkpoint, 'p_scorer'))
        state_dict.update(filter_state_dict(q_scorer_checkpoint, 'q_scorer'))
        return state_dict

    def training_step(self, batch, batch_idx):
        # ['qid': List[int], 'source':List[str], 'target':List[str], 'doc_ids': List[List[int]], 'doc_texts': List[List[str]]]
        output: ELBO = self.loss_fn(batch['source'], batch['target'], batch['doc_texts'])
        log_all(self.expdir, 'train', self.current_epoch, batch_idx, batch, output)
        return output.loss

    def validation_step(self, batch, batch_idx):
        output: ELBO = self.loss_fn(batch['source'], batch['target'], batch['doc_texts'])
        log_all(self.expdir, 'val', self.current_epoch, batch_idx, batch, output)
        return output.loss


    def setup_tsv_files(self):
        with open(Path(self.expdir) / Path(f'p_scores_{self.current_epoch}.tsv'), 'w') as f:
            f.write('stage\tepoch\tq_id\tdoc_id\tp_score\n')
        with open(Path(self.expdir) / Path(f'q_scores_{self.current_epoch}.tsv'), 'w') as f:
            f.write('stage\tepoch\tq_id\tdoc_id\tq_score\n')
        with open(Path(self.expdir) / Path(f'nll_{self.current_epoch}.tsv'), 'w') as f:
            f.write('stage\tepoch\tq_id\tdoc_id\tnll\n')
        with open(Path(self.expdir) / Path(f'metrics_{self.current_epoch}.tsv'), 'w') as f:
            f.write('stage\tepoch\tbatch_idx\tkey\tvalue\n')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

@dataclass
class ReconstructionLoss():
    loss: Union[float, torch.Tensor]
    lm_nll: torch.Tensor
    q_scores: torch.Tensor
    @property
    def metrics(self):
        return [('loss', self.loss)]

    @property
    def intermediate_values(self):
        return [('nll', self.lm_nll), ('q_scores', self.q_scores)]


class ReconstructionLossFn(torch.nn.Module):
    def __init__(self, generator: Generator):
        super().__init__()
        self.generator = generator
        self.generator_nll = LM_NLL(self.generator)

    def forward(self, sources: List[str], targets: List[str], batched_docs: List[List[str]], q_scores: torch.Tensor):
        q_probs = stable_softmax(q_scores, dim=1) #q_scores.shape = n_instances x n_docs
        generator_log_prob = -self.generator_nll(sources, targets, batched_docs) #Shape: n_instances x n_docs

        reconstruction_loss = -(q_probs * generator_log_prob).sum()
        return ReconstructionLoss(reconstruction_loss, -generator_log_prob, q_scores)

class OnlyGeneratorTraining(pl.LightningModule, InheritableCheckpointMixin):
    # We assume that the Q scorer is fixed and hence doesn't need to be run
    def __init__(self, expdir='', lr=1e-3, truncate_query_from_start=False):
        super().__init__()
        self._generator = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
        self._generator_tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        self.generator = Generator(self._generator, self._generator_tokenizer)
        self.lr = lr
        self.expdir = expdir

    def on_train_epoch_start(self):
        self.setup_tsv_files()

    def set_loss_fn(self):
        self.loss_fn = ReconstructionLossFn(self.generator)

    @staticmethod
    def extract_state_dict_from_checkpoints(generator_checkpoint):
        generator_checkpoint = torch.load(generator_checkpoint, torch.device('cpu'))
        state_dict = filter_state_dict(generator_checkpoint, 'generator')
        state_dict.update(filter_state_dict(generator_checkpoint, '_generator'))
        return state_dict

    def training_step(self, batch, batch_idx):
        # ['qid': List[int], 'source':List[str], 'target':List[str], 'doc_ids': List[List[int]], 'doc_texts': List[List[str]]]
        output: ReconstructionLoss = self.loss_fn(batch['source'], batch['target'], batch['doc_texts'], batch['doc_scores'])
        self.my_log('train', batch_idx, batch, output)
        return output.loss

    def validation_step(self, batch, batch_idx):
        output: ReconstructionLoss = self.loss_fn(batch['source'], batch['target'], batch['doc_texts'], batch['doc_scores'])
        self.my_log('val', batch_idx, batch, output)
        return output.loss

    def my_log(self, loop, batch_idx, batch, output):
        for name, value in [('loss', output.loss)]:
            log_value(Path(self.expdir) / Path('metrics.tsv'), loop, self.current_epoch, batch_idx, name, value)

        for fname, values in [('q_scores.tsv', output.q_scores),
                              ('nll.tsv', output.lm_nll)]:
            log_batch_value(Path(self.expdir) / fname, loop, self.current_epoch, batch['qid'], batch['doc_ids'],
                            values)

    def setup_tsv_files(self):
        with open(Path(self.expdir) / Path(f'q_scores_{self.current_epoch}.tsv'), 'w') as f:
            f.write('stage\tepoch\tq_id\tdoc_id\tq_score\n')
        with open(Path(self.expdir) / Path(f'nll_{self.current_epoch}.tsv'), 'w') as f:
            f.write('stage\tepoch\tq_id\tdoc_id\tnll\n')
        with open(Path(self.expdir) / Path(f'metrics_{self.current_epoch}.tsv'), 'w') as f:
            f.write('stage\tepoch\tbatch_idx\tkey\tvalue\n')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

@dataclass
class KLDivergence():
    loss: Union[float, torch.Tensor]
    p_scores: torch.Tensor
    q_scores: torch.Tensor

    @property
    def metrics(self):
        return [('loss', self.loss)]

    @property
    def intermediate_values(self):
        return [('p_scores', self.p_scores), ('q_scores', self.q_scores)]

class KLDivergenceFn(torch.nn.Module):
    def __init__(self, p_scorer: ColBERTScorer):
        super().__init__()
        self.p_scorer = p_scorer

    def forward(self, sources: List[str], batched_docs: List[List[str]], q_scores: torch.Tensor):
        q_probs = stable_softmax(q_scores, dim=1) #q_scores.shape = n_instances x n_docs
        p_scores = self.p_scorer(sources, batched_docs)
        p_probs = stable_softmax(p_scores, dim=1)
        kl_regularization = (q_probs * (q_probs.log() - p_probs.log())).sum()
        return KLDivergence(kl_regularization, p_scores, q_scores)


class OnlyRetrieverTraining(pl.LightningModule, InheritableCheckpointMixin):
    def __init__(self, loss_fn, query_maxlen, doc_maxlen, expdir='', lr=1e-3, truncate_query_from_start=False):
        super().__init__()
        self.p_scorer = ColBERTScorer.from_pretrained('bert-base-uncased',
                                                      truncate_query_from_start = truncate_query_from_start,
                                                      query_maxlen=query_maxlen,
                                                      doc_maxlen=doc_maxlen)
        self.loss_fn_constructor = loss_fn
        self.set_loss_fn()
        self.lr = lr
        self.expdir = expdir

    def set_loss_fn(self):
        self.loss_fn = self.loss_fn_constructor(self.p_scorer)


    @staticmethod
    def extract_state_dict_from_checkpoints(p_scorer_checkpoint):
        p_scorer_checkpoint = torch.load(p_scorer_checkpoint, torch.device('cpu'))
        state_dict = filter_state_dict(p_scorer_checkpoint, 'p_scorer')
        return state_dict

    def training_step(self, batch, batch_idx):
        # ['qid': List[int], 'source':List[str], 'target':List[str], 'doc_ids': List[List[int]], 'doc_texts': List[List[str]]]
        output: KLDivergence = self.loss_fn(batch['source'], batch['doc_texts'], batch['doc_scores'])
        self.my_log('train', batch_idx, batch, output)
        return output.loss

    def validation_step(self, batch, batch_idx):
        output: KLDivergence = self.loss_fn(batch['source'],batch['doc_texts'], batch['doc_scores'])
        self.my_log('val', batch_idx, batch, output)
        return output.loss

    def my_log(self, loop, batch_idx, batch, output):
        for name, value in [('loss', output.loss)]:
            log_value(Path(self.expdir) / Path('metrics.tsv'), loop, self.current_epoch, batch_idx, name, value)

        for fname, values in [('p_scores.tsv', output.p_scores),
                              ('q_scores.tsv', output.q_scores)]:
            log_batch_value(Path(self.expdir) / fname, loop, self.current_epoch, batch['qid'], batch['doc_ids'],
                            values)
    def on_train_epoch_start(self):
        self.setup_tsv_files()

    def setup_tsv_files(self):
        with open(Path(self.expdir) / Path(f'p_scores_{self.current_epoch}.tsv'), 'w') as f:
            f.write('stage\tepoch\tq_id\tdoc_id\tp_score\n')
        with open(Path(self.expdir) / Path(f'q_scores_{self.current_epoch}.tsv'), 'w') as f:
            f.write('stage\tepoch\tq_id\tdoc_id\tq_score\n')
        with open(Path(self.expdir) / Path(f'metrics_{self.current_epoch}.tsv'), 'w') as f:
            f.write('stage\tepoch\tbatch_idx\tkey\tvalue\n')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

def filter_state_dict(ckpt, key):
    return {k: v for k, v in ckpt['state_dict'].items() if k.startswith(key)}


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Script to jointly train retriever and generator')
    base_path = Path('/u/scr/ashwinp/research/readi')
    rerank_exp_base_path = Path('/scr/biggest/ashwinp/experiments/colbert-rerank/')
    scorer_group = parser.add_argument_group(title='scorer (ColBERT) args')
    scorer_group.add_argument('--query_maxlen', dest='query_maxlen', default=64, type=int)
    scorer_group.add_argument('--doc_maxlen', dest='doc_maxlen', default=180, type=int)
    scorer_group.add_argument('--truncate_query_from_start', action='store_true', default=False)

    checkpoints_group = parser.add_argument_group(title='paths to various checkpoints')
    checkpoints_group.add_argument('--p_scorer_checkpoint', type=str, help="Path to p_scorer checkpoint, can be from colbert or qtraining"), #default='/scr/biggest/ashwinp/readi/checkpoints/colbert/colbert-400000.dnn')
    checkpoints_group.add_argument('--q_scorer_checkpoint', type=str, help="Path to q_scorer checkpoint, can be from colbert or qtraining"), #default='/scr/biggest/ashwinp/readi/checkpoints/colbert/colbert-400000.dnn')
    checkpoints_group.add_argument('--scorer_checkpoint_type', type=str, help="The scorer checkpoints were generated by: colbert, qtraining")
    checkpoints_group.add_argument('--generator_checkpoint', default=None, type=str, help='Path to generator checkpoint')
    checkpoints_group.add_argument('--resume_training_from_checkpoint', type=str, help="Optional, if given, loads the scorers and generator from the checkpoint. Resumes training using the resumed pytorch lightning trainer.resumed ")

    paths_group = parser.add_argument_group(title='input file paths')
    paths_group.add_argument('--train_source_path', type=str, default=(base_path / 'data/nq/train.source').as_posix(),
                        help='Path to train.source file, each line contains input to the generator')
    paths_group.add_argument('--train_target_path', type=str, default=(base_path / 'data/nq/train.target').as_posix(),
                        help='Path to train.target file, each line contains expected output from the generator')
    paths_group.add_argument('--train_p_ranked_passages', type=str, default=(rerank_exp_base_path / '10/ranking_passages.tsv').as_posix() ,
                        help='Path to ranking_passages.tsv, retrieved and ranked using p-scorer')
    paths_group.add_argument('--train_q_ranked_passages', type=str, default=(rerank_exp_base_path / '11/ranking_passages.tsv' ).as_posix(),
                        help='Path to ranking_passages.tsv, retrieved and ranked using q-scorer')

    paths_group.add_argument('--val_source_path', type=str, default=(base_path / 'data/nq/val.source').as_posix(),
                             help='Path to train.source file, each line contains input to the generator')
    paths_group.add_argument('--val_target_path', type=str, default=(base_path / 'data/nq/val.target').as_posix(),
                             help='Path to train.target file, each line contains expected output from the generator')
    paths_group.add_argument('--val_p_ranked_passages', type=str, default=(rerank_exp_base_path / '16/ranking_passages.tsv').as_posix() ,
                             help='Path to ranking_passages.tsv, retrieved and ranked using p-scorer')
    paths_group.add_argument('--val_q_ranked_passages', type=str, default=(rerank_exp_base_path / '17/ranking_passages.tsv').as_posix() ,
                             help='Path to ranking_passages.tsv, retrieved and ranked using q-scorer')

    training_args_group = parser.add_argument_group(title='training args')
    training_args_group.add_argument('--batch_size', type=int, default=3, help='training batch size')
    training_args_group.add_argument('--loss_type', type=str,
                                     help='Training loss to use. Choices: [NLL, Marginalized, ELBO, Reconstruction, KLD, PosNeg]')
    training_args_group.add_argument('--n_sampled_docs_train', type=int, default=8,
                                     help="Number of docs to sample for each instance (Marginalized, ELBO)")
    training_args_group.add_argument('--n_sampled_docs_valid', type=int, default=100,
                                     help="Number of docs to sample for each validation instance (Marginalized, ELBO)")
    training_args_group.add_argument('--docs_top_k', type=int, default=100,
                                     help="Sample from top_k docs (Marginalized, ELBO)")
    training_args_group.add_argument('--docs_sampling_temperature', type=float, default=1,
                                     help="Temperature used for sampling docs (Marginalized, ELBO)")
    training_args_group.add_argument('--lr', type=float, default=1e-6, help='Adam\'s Learning rate')
    training_args_group.add_argument('--accumulate_grad_batches', type=int, default=16, help='Accumulate gradients for given number of batches')
    training_args_group.add_argument('--gpus', type=int, default=1, help='Number of gpus to use')
    training_args_group.add_argument('--doc_sampler', type=str,
                                     help='Sampler to use during training: {SimpleDocumentSampler(Marginalized), GuidedDocumentSampler(ELBO), GuidedNoIntersectionSampler(ELBO), RankPNDocumentSampler(ELBO)}, PosteriorDocumentSampler(Reconstruction)')
    training_args_group.add_argument('--max_epochs', type=int, default=10, help="Trainer stops training after max_epochs")
    training_args_group.add_argument('--limit_train_batches', default=1.0, type=int, help="Limits number of training batches per epoch. Workaround for some bug where skipped instances reduces number of batches leading pytorch lightning to not detect end of epoch")
    training_args_group.add_argument('--limit_val_batches', default=1.0, type=int, help="Limits number of validation batches per epoch.")


    Experiment.add_argument_group(parser)
    args = parser.parse_args()
    if args.resume_training_from_checkpoint:
        print(f"Resuming all internal models from {args.resume_training_from_checkpoint}")
        args.p_scorer_checkpoint = args.resume_training_from_checkpoint
        args.q_scorer_checkpoint = args.resume_training_from_checkpoint
        args.generator_checkpoint = args.resume_training_from_checkpoint
        args.scorer_checkpoint_type = 'qtraining'

    if node_rank ==0 and local_rank == 0:
        experiment = Experiment.from_parser(parser)
        curexpdir = experiment.curexpdir
    else:
        curexpdir = os.path.join(args.experiments_directory, args.experiment_id)

    # TODO: Currently only using local rank and args.ngpus
    # The code won't work with multiple nodes, for which one needs to have a global rank and world size
    logger = CSVLogger(save_dir=args.experiments_directory, name='', version=args.experiment_id)
    checkpoint_callback = ModelCheckpoint(monitor=None, save_top_k=-1)
    if args.resume_training_from_checkpoint:
        print("Overriding the model using the checkpoint")
        trainer = Trainer(gpus=args.gpus, logger=logger,
                          default_root_dir=curexpdir, track_grad_norm=2,
                          accumulate_grad_batches=args.accumulate_grad_batches, accelerator='ddp', max_epochs=args.max_epochs, callbacks=[checkpoint_callback], resume_from_checkpoint=args.resume_from_checkpoint, limit_train_batches=args.limit_train_batches, limit_val_batches=args.limit_val_batches)
        trainer.max_epochs = trainer.current_epoch+args.max_epochs

        #if args.loss_type == 'NLL':
        #    model = NLLLossSystem.load_from_checkpoint(checkpoint_path=args.resume_from_checkpoint,
        #                                               lr=args.lr, expdir=curexpdir,
        #                                               truncate_query_from_start=args.truncate_query_from_start)
        #elif args.loss_type == 'Marginalized':
        #    model = MarginalizedLossSystem.load_from_checkpoint(checkpoint_path=args.resume_from_checkpoint,
        #                                                        p_scorer_checkpoint= args.p_scorer_checkpoint,
        #                                                        query_maxlen=args.query_maxlen,
        #                                                        doc_maxlen=args.doc_maxlen,
        #                                                        expdir=curexpdir, lr=args.lr,
        #                                                        truncate_query_from_start=args.truncate_query_from_start)
        #elif args.loss_type == 'ELBO':
        #    model = ELBOLossSystem.load_from_checkpoint(checkpoint_path=args.resume_from_checkpoint,
        #                                                p_scorer_checkpoint=args.p_scorer_checkpoint,
        #                                                q_scorer_checkpoint=args.p_scorer_checkpoint,
        #                                                query_maxlen=args.query_maxlen,
        #                                                doc_maxlen=args.doc_maxlen, expdir=curexpdir, lr=args.lr,
        #                                                truncate_query_from_start=args.truncate_query_from_start)


    else:
        trainer = Trainer(gpus=args.gpus, logger=logger,
                          default_root_dir=curexpdir, track_grad_norm=2,
                          accumulate_grad_batches=args.accumulate_grad_batches, accelerator='ddp', max_epochs=args.max_epochs, callbacks=[checkpoint_callback], limit_train_batches=args.limit_train_batches, limit_val_batches=args.limit_val_batches)

    # Create models
    if args.loss_type == 'NLL':
        # Still old style
        model = NLLLossSystem(lr=args.lr, expdir=curexpdir,
                              truncate_query_from_start=args.truncate_query_from_start)

    elif args.loss_type == 'Marginalized':
        # Still old style loading from checkpoints
        #TODO, test if the following are identical
        model = MarginalizedLossSystem(args.query_maxlen, args.doc_maxlen,
                                       expdir=curexpdir, lr=args.lr,
                                       truncate_query_from_start=args.truncate_query_from_start)
        if args.scorer_checkpoint_type == 'colbert':
            state_dict = MarginalizedLossSystem.extract_state_dict_from_colbert_checkpoints(
                p_scorer_checkpoint=args.p_scorer_checkpoint)
        elif args.scorer_checkpoint_type == 'qtraining':
            state_dict = MarginalizedLossSystem.extract_state_dict_from_checkpoints(p_scorer_checkpoint=args.p_scorer_checkpoint,
                                                                           generator_checkpoint=args.generator_checkpoint)
        else:
            assert False
        model = MarginalizedLossSystem.init_from_checkpoints(state_dict, query_maxlen=args.query_maxlen,
                                                     doc_maxlen=args.doc_maxlen, expdir=curexpdir, lr=args.lr,
                                                     truncate_query_from_start=args.truncate_query_from_start)
    elif args.loss_type == 'ELBO':
        #TODO, test if the following are identical
        model = ELBOLossSystem(args.query_maxlen, args.doc_maxlen, expdir=curexpdir, lr=args.lr, truncate_query_from_start=args.truncate_query_from_start, p_scorer_checkpoint=args.p_scorer_checkpoint, q_scorer_checkpoint=args.q_scorer_checkpoint, )
        if args.scorer_checkpoint_type == 'colbert':
            state_dict = ELBOLossSystem.extract_state_dict_from_colbert_checkpoints(
                p_scorer_checkpoint=args.p_scorer_checkpoint, q_scorer_checkpoint=args.q_scorer_checkpoint)
        elif args.scorer_checkpoint_type == 'qtraining':
            state_dict = ELBOLossSystem.extract_state_dict_from_checkpoints(p_scorer_checkpoint=args.p_scorer_checkpoint,
                                                                           q_scorer_checkpoint=args.q_scorer_checkpoint,
                                                                           generator_checkpoint=args.generator_checkpoint)
        else:
            assert False

        model = ELBOLossSystem.init_from_checkpoints(state_dict, query_maxlen=args.query_maxlen, doc_maxlen=args.doc_maxlen,
                                                     expdir=curexpdir, lr=args.lr,
                                                     truncate_query_from_start=args.truncate_query_from_start)
    elif args.loss_type == 'Reconstruction':
        state_dict = OnlyGeneratorTraining.extract_state_dict_from_checkpoints(generator_checkpoint=args.generator_checkpoint)
        model = OnlyGeneratorTraining.init_from_checkpoints(state_dict,  expdir=curexpdir, lr=args.lr, truncate_query_from_start=args.truncate_query_from_start )
    elif args.loss_type == 'KLD' or args.loss_type=='PosNeg':
        if args.loss_type == 'KLD':
            loss_fn = KLDivergenceFn
        elif args.loss_type == 'PosNeg':
            loss_fn = None
        else:
            raise False
        if args.scorer_checkpoint_type == 'qtraining':
            state_dict = OnlyRetrieverTraining.extract_state_dict_from_checkpoints(
                p_scorer_checkpoint=args.p_scorer_checkpoint)
        else:
            assert False
        model = OnlyRetrieverTraining.init_from_checkpoints(state_dict, loss_fn=loss_fn, query_maxlen=args.query_maxlen,
                                                     doc_maxlen=args.doc_maxlen, expdir=curexpdir, lr=args.lr,
                                                        truncate_query_from_start=args.truncate_query_from_start)
    else:
            assert False, "loss_type not in {NLL, Marginalized, ELBO, Reconstruction, KLD, PosNeg}"

    secondary_training = args.loss_type in {'Reconstruction', 'KLD', 'PosNeg'}
    if args.loss_type == 'NLL':
        train_dataset = Seq2SeqDataset(args.train_source_path, args.train_target_path, worker_id=local_rank, n_workers=args.gpus)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size)
        val_dataset = Seq2SeqDataset(args.val_source_path, args.val_target_path, worker_id=local_rank, n_workers=args.gpus)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)
    elif args.loss_type == 'Marginalized':
        assert args.doc_sampler == 'SimpleDocumentSampler'
        doc_sampler = SimpleDocumentSampler(args.n_sampled_docs_train, temperature=args.docs_sampling_temperature, top_k=args.docs_top_k)
        train_dataset = PDataset(args.train_source_path, args.train_target_path, args.train_p_ranked_passages, doc_sampler, worker_id=local_rank, n_workers=args.gpus)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
        val_doc_sampler = SimpleDocumentSampler(args.n_sampled_docs_valid, temperature=args.docs_sampling_temperature, top_k=args.docs_top_k)
        val_dataset = PDataset(args.val_source_path, args.val_target_path, args.val_p_ranked_passages, val_doc_sampler, worker_id=local_rank, n_workers=args.gpus)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    elif args.loss_type == 'Reconstruction':
        assert args.doc_sampler in {'PosteriorDocumentSampler'}
        doc_sampler = PosteriorDocumentSampler(args.n_sampled_docs_train, top_k=args.docs_top_k)
        train_dataset = PQDataset(args.train_source_path, args.train_target_path, args.train_p_ranked_passages,
                                  args.train_q_ranked_passages, doc_sampler, worker_id=local_rank, n_workers=args.gpus,
                                  yield_scores=secondary_training, include_unrelated=False)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
        val_dataset = PQDataset(args.val_source_path, args.val_target_path, args.val_p_ranked_passages,
                                args.val_q_ranked_passages, doc_sampler, worker_id = local_rank, n_workers = args.gpus,
                                yield_scores = secondary_training, include_unrelated=False)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    elif args.loss_type in {'KLD'}:
        assert args.doc_sampler in {'GuidedDocumentSampler', 'RankPNDocumentSampler', 'PosteriorDocumentSampler'}
        if args.doc_sampler == 'GuidedDocumentSampler':
            doc_sampler = GuidedDocumentSampler(args.n_sampled_docs_train, temperature=args.docs_sampling_temperature, top_k=args.docs_top_k, )
        elif args.doc_sampler == 'RankPNDocumentSampler':
            doc_sampler = RankPNDocumentSampler(args.n_sampled_docs_train)
        elif args.doc_sampler == 'PosteriorDocumentSampler':
            doc_sampler = PosteriorDocumentSampler(args.n_sampled_docs_train, top_k=args.docs_top_k)
        else:
            assert False
        train_dataset = PQDataset(args.train_source_path, args.train_target_path, args.train_p_ranked_passages,
                                  args.train_q_ranked_passages, doc_sampler, worker_id=local_rank, n_workers=args.gpus,
                                  yield_scores=secondary_training)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
        val_dataset = PQDataset(args.val_source_path, args.val_target_path, args.val_p_ranked_passages,
                                args.val_q_ranked_passages, doc_sampler, worker_id = local_rank, n_workers = args.gpus,
                                yield_scores = secondary_training)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn)

    elif args.loss_type in {'ELBO',  'PosNeg'} :
        assert args.doc_sampler in {'GuidedDocumentSampler', 'GuidedNoIntersectionDocumentSampler', 'RankPNDocumentSampler', 'PosteriorDocumentSampler'}
        if args.doc_sampler == 'GuidedDocumentSampler':
            doc_sampler = GuidedDocumentSampler(args.n_sampled_docs_train, temperature=args.docs_sampling_temperature, top_k=args.docs_top_k, )
        elif args.doc_sampler == 'GuidedNoIntersectionDocumentSampler':
            doc_sampler = GuidedNoIntersectionDocumentSampler(args.n_sampled_docs_train, temperature=args.docs_sampling_temperature, top_k=args.docs_top_k)
        elif args.doc_sampler == 'RankPNDocumentSampler':
            doc_sampler = RankPNDocumentSampler(args.n_sampled_docs_train)
        elif args.doc_sampler == 'PosteriorDocumentSampler':
            doc_sampler = PosteriorDocumentSampler(args.n_sampled_docs_train, top_k=args.docs_top_k)
        else:
            assert False

        train_dataset = PQDataset(args.train_source_path, args.train_target_path, args.train_p_ranked_passages,
                                  args.train_q_ranked_passages, doc_sampler, worker_id=local_rank, n_workers=args.gpus,
                                  yield_scores=secondary_training)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
        val_doc_sampler = SimpleDocumentSampler(args.n_sampled_docs_valid,
                                                temperature=args.docs_sampling_temperature, top_k=args.docs_top_k)
        val_dataset = PDataset(args.val_source_path, args.val_target_path, args.val_p_ranked_passages, val_doc_sampler,
                               worker_id=local_rank, n_workers=args.gpus)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn)

    #trainer = Trainer(gpus=args.gpus, logger=logger, default_root_dir=curexpdir, track_grad_norm=2,
    #                  accumulate_grad_batches=args.accumulate_grad_batches, fast_dev_run=True)#, callbacks=[checkpoint_callback])
    #trainer.fit(model, train_dataloader, val_dataloader)
    trainer.fit(model, train_dataloader, val_dataloader)


