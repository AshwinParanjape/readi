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
        return self.score(self.query(*Q), self.doc(*D))

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


class DocumentSampler:
    def __call__(self, retrievals):
        raise NotImplementedError("Sampler needs to implement __call__method")

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
class GuidedDocumentSampler(DocumentSampler):
    def __init__(self, n, temperature=1, top_k=None):
        self.n = n
        self.temperature=temperature
        assert n<=top_k, f"top_k={top_k} should at least be n={n}"
        self.top_k = top_k
        self.p_intersection_q_sampler = SimpleDocumentSampler(self.n//2, self.temperature, self.top_k)
        self.p_minus_q_sampler = SimpleDocumentSampler(self.n//4, self.temperature, self.top_k)
        self.q_minus_p_sampler = SimpleDocumentSampler(self.n//4, self.temperature, self.top_k)

    def __call__(self, retrievals: pd.DataFrame):
        # retrievals has columns ['qid', 'pid', 'p_score', 'score_q', 'doc_text', 'title', 'text']
        p_intersection_q = retrievals[(retrievals['score_p'].notna()) & (retrievals['score_q'].notna())].copy()
        p_intersection_q['score'] = p_intersection_q['score_q']
        p_intersection_q_samples = self.p_intersection_q_sampler(p_intersection_q)

        p_minus_q = retrievals[(retrievals['score_p'].notna()) & (retrievals['score_q'].isna())].copy()
        p_minus_q['score'] = p_minus_q['score_p']
        p_minus_q_samples = self.p_minus_q_sampler(p_minus_q)

        q_minus_p = retrievals[(retrievals['score_q'].notna()) & (retrievals['score_p'].isna())].copy()
        q_minus_p['score'] = q_minus_p['score_q']
        q_minus_p_samples = self.q_minus_p_sampler(q_minus_p)

        mixed_samples = pd.concat([p_intersection_q_samples, p_minus_q_samples, q_minus_p_samples])

        # If not enough samples were gotten (because some of the three sets not containing enough to sample from)
        # Add a few more on an ad-hoc basis
        if len(mixed_samples) < self.n:
            diff = self.n - len(mixed_samples)
            q_docs = retrievals[(retrievals['score_q'].notna())].copy()
            q_docs['score'] = q_docs['score_q']
            extra_samples = SimpleDocumentSampler(diff, self.temperature, self.top_k)(q_docs)
            mixed_samples = pd.concat([mixed_samples, extra_samples])

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


class PDataset(torch.utils.data.IterableDataset):
    def __init__(self, source_path: str, target_path: str, p_retrievals_path: str, sampler:DocumentSampler, worker_id=0, n_workers=1):
        self.source = pd.read_csv(source_path, sep='\t', names=['source'], dtype=str, na_filter=False)
        if target_path:
            self.target = pd.read_csv(target_path, sep='\t', names=['target'], dtype=str, na_filter=False)
        else:
            self.target = pd.DataFrame()
            self.target['target'] = self.source['source'] # To quickly get the same shape
            self.target['target'] = ''
        self.p_retrievals = ClosedSetRetrievals(p_retrievals_path)
        self.cached_scores: Dict[int, Dict[int, float]] = defaultdict(dict)
        self.sampler = sampler
        self.worker_id = worker_id
        self.n_workers = n_workers

    def __iter__(self):
        for qid, (source, target, (p_qid, p_retrievals)) in enumerate(zip(self.source['source'], self.target['target'], self.p_retrievals)):
            #assert (qid == p_qid) , (qid, p_qid)
            if qid % self.n_workers == self.worker_id and qid < len(self)*self.n_workers:  # This query belongs to this worker
                sampled_retrievals = self.sampler(p_retrievals)
                yield {'qid': qid,
                        'source': source,
                        'target': target,
                        'doc_ids': sampled_retrievals['pid'].tolist(),
                        'doc_texts': sampled_retrievals['text'].tolist(),
                        }
    def __len__(self):
        return len(self.source)//self.n_workers


class PQDataset(torch.utils.data.IterableDataset):
    def __init__(self, source_path:str, target_path: str, p_retrievals_path: str, q_retrievals_path: str, sampler: DocumentSampler, worker_id=0,n_workers=1):
        self.source = pd.read_csv(source_path, sep='\t', names=['source'], dtype=str, na_filter=False)
        self.target = pd.read_csv(target_path, sep='\t', names=['target'], dtype=str, na_filter=False)
        self.p_retrievals = ClosedSetRetrievals(p_retrievals_path)
        self.q_retrievals = ClosedSetRetrievals(q_retrievals_path)
        self.sampler = sampler
        self.worker_id = worker_id
        self.n_workers = n_workers

    def __iter__(self):
        for qid, (source, target, (p_qid, p_retrievals), (q_qid, q_retrievals)) in enumerate(zip(self.source['source'], self.target['target'], self.p_retrievals, self.q_retrievals)):
            #assert (qid == p_qid) and (qid == q_qid), (qid, p_qid, q_qid)
            if qid % self.n_workers == self.worker_id and qid < len(self)*self.n_workers:  # This query belongs to this worker
                merged_retrievals = p_retrievals.merge(q_retrievals, how='outer', on=['qid', 'pid', 'doc_text', 'title', 'text'], suffixes = ('_p', '_q'))
                sampled_retrievals = self.sampler(merged_retrievals)
                yield {'qid': qid,
                        'source': source,
                        'target': target,
                        'doc_ids': sampled_retrievals['pid'].tolist(),
                        'doc_texts': sampled_retrievals['text'].tolist()
                 }

    def __len__(self):
        return len(self.source)//self.n_workers


# TODO: override collate function to simply collate tuples into a list
def collate_fn(batch: Dict):
    collated = default_collate(
        [{k:v for k, v in d.items() if k in {'qid', 'source', 'target'}} for d in batch]
    )
    collated['doc_ids'] = [d['doc_ids'] for d in batch ]
    collated['doc_texts'] = [d['doc_texts'] for d in batch ]
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
        output = self.generator.generate(input_ids=input_encoding['input_ids'],
                                        attention_mask=input_encoding['attention_mask'],
                                         return_dict_in_generate=True, output_scores=True, do_sample=True,
                                        **generation_kwargs)
        decoded_output = self.tokenizer.batch_decode(output.sequences)
        output.strings = decoded_output
        return output

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
        query_embeds = self.query(*Q)
        doc_embeds = self.doc(*D)
        doc_embeds = doc_embeds.view(n_instances, n_docs, *doc_embeds.shape[1:])
        score = self.score(query_embeds, doc_embeds)
        return score

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
        loss = -torch.logsumexp(doc_log_probs + generator_log_prob, dim=(0, 1))
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

        marginalized_nll_loss = -torch.logsumexp(p_log_probs + generator_log_prob, dim=(0, 1))

        reconstruction_score = (q_probs * generator_log_prob).sum()
        kl_regularization = (q_probs * (q_probs.log() - p_probs.log())).sum()
        elbo_loss = -(reconstruction_score - kl_regularization)

        return ELBO(elbo_loss, reconstruction_score, kl_regularization, marginalized_nll_loss, -generator_log_prob, p_scores, q_scores)

class NLLLossSystem(pl.LightningModule):
    def __init__(self, lr=1e-3, truncate_query_from_start=False) :
        super().__init__()
        self._generator = BartForConditionalGeneration.from_pretrained("facebook/bart-base", force_bos_token_to_be_generated=True)
        self._generator_tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        #self.generator = Generator(self._generator, self._generator_tokenizer, truncate_from_start=truncate_query_from_start)
        self.generator = Generator(self._generator, self._generator_tokenizer)
        self.loss_fn = LM_NLL(self.generator)
        self.lr = lr
        with open(Path(self.expdir) / Path('metrics.tsv'), 'w') as f:
            f.write('stage\tepoch\tbatch_idx\tkey\tvalue\n')

    def training_step(self, batch, batch_idx):
        # ['qid': List[int], 'source':List[str], 'target':List[str], 'doc_ids': List[List[int]], 'doc_texts': List[List[str]]]
        output = self.loss_fn(batch['source'], batch['target'])

        log_value(Path(self.expdir)/ Path('metrics.tsv'), 'train', self.current_epoch, batch_idx, 'loss', output.sum())

        return output.sum()

    def training_step(self, batch, batch_idx):
        # ['qid': List[int], 'source':List[str], 'target':List[str], 'doc_ids': List[List[int]], 'doc_texts': List[List[str]]]
        output = self.loss_fn(batch['source'], batch['target'])

        log_value(Path(self.expdir)/ Path('metrics.tsv'), 'val', self.current_epoch, batch_idx, 'loss', output.sum())

        return output.sum()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

class MarginalizedLossSystem(pl.LightningModule):
    def __init__(self, p_scorer_checkpoint, query_maxlen, doc_maxlen, expdir='', lr=1e-3, truncate_query_from_start=False) :
        super().__init__()
        self._generator = BartForConditionalGeneration.from_pretrained("facebook/bart-base", force_bos_token_to_be_generated=True)
        self._generator_tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        #self._generator_tokenizer.add_tokens([DOC_TOKEN, TEXT_TOKEN])
        #self.generator = Generator(self._generator, self._generator_tokenizer, truncate_from_start=truncate_query_from_start)
        self.generator = Generator(self._generator, self._generator_tokenizer)
        self.p_scorer = ColBERTScorer.from_pretrained('bert-base-uncased',
                                          truncate_query_from_start = truncate_query_from_start,
                                          query_maxlen=query_maxlen,
                                          doc_maxlen=doc_maxlen,
                                          )
        saved_state_dict = torch.load(p_scorer_checkpoint, map_location='cpu')
        self.p_scorer.load_state_dict(saved_state_dict['model_state_dict'], strict=False)
        self.loss_fn = MarginalizedNLLFn(self.p_scorer, self.generator)

        self.expdir = expdir
        self.lr = lr
        with open(Path(self.expdir) / Path('p_scores.tsv'), 'w') as f:
            f.write('stage\tepoch\tq_id\tdoc_id\tp_score\n')
        with open(Path(self.expdir) / Path('nll.tsv'), 'w') as f:
            f.write('stage\tepoch\tq_id\tdoc_id\tnll\n')
        with open(Path(self.expdir) / Path('metrics.tsv'), 'w') as f:
            f.write('stage\tepoch\tbatch_idx\tkey\tvalue\n')

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

class ELBOLossSystem(pl.LightningModule):
    def __init__(self, p_scorer_checkpoint, q_scorer_checkpoint, query_maxlen, doc_maxlen, expdir='', lr=1e-3, truncate_query_from_start=False):
        super().__init__()
        self._generator = BartForConditionalGeneration.from_pretrained("facebook/bart-base", force_bos_token_to_be_generated=True)
        self._generator_tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        #self._generator_tokenizer.add_tokens([DOC_TOKEN, TEXT_TOKEN])
        #self.generator = Generator(self._generator, self._generator_tokenizer, truncate_from_start=truncate_query_from_start)
        self.generator = Generator(self._generator, self._generator_tokenizer)
        self.p_scorer = ColBERTScorer.from_pretrained('bert-base-uncased',
                                          truncate_query_from_start = truncate_query_from_start,
                                          query_maxlen=query_maxlen,
                                          doc_maxlen=doc_maxlen)
        saved_state_dict = torch.load(p_scorer_checkpoint, map_location='cpu')
        self.p_scorer.load_state_dict(saved_state_dict['model_state_dict'], strict=False)

        self.q_scorer = ColBERTScorer.from_pretrained('bert-base-uncased',
                                          truncate_query_from_start = truncate_query_from_start,
                                          query_maxlen=query_maxlen,
                                          doc_maxlen=doc_maxlen)
        saved_state_dict = torch.load(q_scorer_checkpoint, map_location='cpu')
        self.q_scorer.load_state_dict(saved_state_dict['model_state_dict'], strict=False)
        self.loss_fn = ELBOFn(self.p_scorer, self.q_scorer, self.generator)
        self.lr = lr
        self.expdir = expdir
        with open(Path(self.expdir)/ Path('p_scores.tsv'), 'w') as f:
            f.write('stage\tepoch\tq_id\tdoc_id\tp_score\n')
        with open(Path(self.expdir)/ Path('q_scores.tsv'), 'w') as f:
            f.write('stage\tepoch\tq_id\tdoc_id\tq_score\n')
        with open(Path(self.expdir)/Path('nll.tsv'), 'w') as f:
            f.write('stage\tepoch\tq_id\tdoc_id\tnll\n')
        with open(Path(self.expdir)/Path('metrics.tsv'), 'w') as f:
            f.write('stage\tepoch\tbatch_idx\tkey\tvalue\n')


    def training_step(self, batch, batch_idx):
        # ['qid': List[int], 'source':List[str], 'target':List[str], 'doc_ids': List[List[int]], 'doc_texts': List[List[str]]]
        output: ELBO = self.loss_fn(batch['source'], batch['target'], batch['doc_texts'])
        log_value(Path(self.expdir)/ Path('metrics.tsv'), 'train', self.current_epoch, batch_idx, 'loss', output.loss)
        log_value(Path(self.expdir)/ Path('metrics.tsv'), 'train', self.current_epoch, batch_idx, 'marginalized_loss',  output.marginalized_loss)
        log_value(Path(self.expdir)/ Path('metrics.tsv'), 'train', self.current_epoch, batch_idx, 'reconstruction_score', output.reconstruction_score)
        log_value(Path(self.expdir)/ Path('metrics.tsv'), 'train', self.current_epoch, batch_idx, 'kl_divergence',  output.kl_divergence)

        log_batch_value(Path(self.expdir)/ Path('p_scores.tsv'), 'train', self.current_epoch, batch['qid'], batch['doc_ids'], output.p_scores)
        log_batch_value(Path(self.expdir)/ Path('q_scores.tsv'), 'train', self.current_epoch, batch['qid'], batch['doc_ids'], output.q_scores)
        log_batch_value(Path(self.expdir)/ Path('nll.tsv'), 'train', self.current_epoch, batch['qid'], batch['doc_ids'], output.lm_nll)
        return output.loss

    def validation_step(self, batch, batch_idx):
        output: ELBO = self.loss_fn(batch['source'], batch['target'], batch['doc_texts'])
        log_value(Path(self.expdir)/ Path('metrics.tsv'), 'val', self.current_epoch, batch_idx, 'loss', output.loss)
        log_value(Path(self.expdir)/ Path('metrics.tsv'), 'val', self.current_epoch, batch_idx, 'marginalized_loss',  output.marginalized_loss)
        log_value(Path(self.expdir)/ Path('metrics.tsv'), 'val', self.current_epoch, batch_idx, 'reconstruction_score', output.reconstruction_score)
        log_value(Path(self.expdir)/ Path('metrics.tsv'), 'val', self.current_epoch, batch_idx, 'kl_divergence',  output.kl_divergence)

        log_batch_value(Path(self.expdir)/ Path('p_scores.tsv'), 'val', self.current_epoch, batch['qid'], batch['doc_ids'], output.p_scores)
        log_batch_value(Path(self.expdir)/ Path('q_scores.tsv'), 'val', self.current_epoch, batch['qid'], batch['doc_ids'], output.q_scores)
        log_batch_value(Path(self.expdir)/ Path('nll.tsv'), 'val', self.current_epoch, batch['qid'], batch['doc_ids'], output.lm_nll)
        return output.loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

def log_value(filename, stage, epoch, batch_idx, key, value):
    with open(filename, 'a') as f:
        f.write(f'{stage}\t{epoch}\t{batch_idx}\t{key}\t{value}\n')

def log_batch_value(filename, stage, epoch, qids, batched_doc_ids, batched_values):
    with open(filename, 'a') as f:
        for qid, doc_ids, values in zip(qids, batched_doc_ids, batched_values):
            for doc_id, value in zip(doc_ids, values):
                f.write(f'{stage}\t{epoch}\t{qid}\t{doc_id}\t{value}\n')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Script to jointly train retriever and generator')
    base_path = Path('/u/scr/ashwinp/research/readi')
    rerank_exp_base_path = Path('/scr/biggest/ashwinp/experiments/colbert-rerank/')
    scorer_group = parser.add_argument_group(title='scorer (ColBERT) args')
    scorer_group.add_argument('--p_scorer_checkpoint', default='/scr/biggest/ashwinp/readi/checkpoints/colbert/colbert-400000.dnn')
    scorer_group.add_argument('--query_maxlen', dest='query_maxlen', default=64, type=int)
    scorer_group.add_argument('--doc_maxlen', dest='doc_maxlen', default=180, type=int)
    scorer_group.add_argument('--truncate_query_from_start', action='store_true', default=False)

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
    training_args_group.add_argument('--loss_type', type=str, default='ELBO',
                                     help='Training loss to use. Choices: [NLL, Marginalized, ELBO]')
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
    training_args_group.add_argument('--doc_sampler', type=str, default='GuidedDocumentSampler',
                                     help='Sampler to use during training: {SimpleDocumentSampler(Marginalized), GuidedDocumentSampler(ELBO), GuidedNoIntersectionSampler(ELBO)}')
    training_args_group.add_argument('--max_epochs', type=int, default=10, help="Trainer stops training after max_epochs")


    Experiment.add_argument_group(parser)
    args = parser.parse_args()
    if node_rank ==0 and local_rank == 0:
        experiment = Experiment.from_parser(parser)
        curexpdir = experiment.curexpdir
    else:
        curexpdir = os.path.join(args.experiments_directory, args.experiment_id)

    # TODO: Currently only using local rank and args.ngpus
    # The code won't work with multiple nodes, for which one needs to have a global rank and world size


    if args.loss_type == 'NLL':
        model = NLLLossSystem(lr = args.lr, expdir=curexpdir, truncate_query_from_start=args.truncate_query_from_start)
        train_dataset = Seq2SeqDataset(args.train_source_path, args.train_target_path, worker_id=local_rank, n_workers=args.gpus)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size)
        val_dataset = Seq2SeqDataset(args.val_source_path, args.val_target_path, worker_id=local_rank, n_workers=args.gpus)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)
    elif args.loss_type == 'Marginalized':
        assert args.doc_sampler == 'SimpleDocumentSampler'
        model = MarginalizedLossSystem(args.p_scorer_checkpoint, args.query_maxlen, args.doc_maxlen, expdir=curexpdir, lr=args.lr, truncate_query_from_start=args.truncate_query_from_start)
        doc_sampler = SimpleDocumentSampler(args.n_sampled_docs_train, temperature=args.docs_sampling_temperature, top_k=args.docs_top_k)
        train_dataset = PDataset(args.train_source_path, args.train_target_path, args.train_p_ranked_passages, doc_sampler, worker_id=local_rank, n_workers=args.gpus)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
        val_doc_sampler = SimpleDocumentSampler(args.n_sampled_docs_valid, temperature=args.docs_sampling_temperature, top_k=args.docs_top_k)
        val_dataset = PDataset(args.val_source_path, args.val_target_path, args.val_p_ranked_passages, val_doc_sampler, worker_id=local_rank, n_workers=args.gpus)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    elif args.loss_type == 'ELBO':
        assert args.doc_sampler == 'GuidedDocumentSampler' or args.doc_sampler == 'GuidedNoIntersectionDocumentSampler'
        model = ELBOLossSystem(args.p_scorer_checkpoint, args.p_scorer_checkpoint, args.query_maxlen, args.doc_maxlen, expdir=curexpdir, lr=args.lr, truncate_query_from_start=args.truncate_query_from_start)
        if args.doc_sampler == 'GuidedDocumentSampler':
            doc_sampler = GuidedDocumentSampler(args.n_sampled_docs_train, temperature=args.docs_sampling_temperature, top_k=args.docs_top_k)
        elif args.doc_sampler == 'GuidedNoIntersectionDocumentSampler':
            doc_sampler = GuidedNoIntersectionDocumentSampler(args.n_sampled_docs_train, temperature=args.docs_sampling_temperature, top_k=args.docs_top_k)
        else:
            assert False

        train_dataset = PQDataset(args.train_source_path, args.train_target_path, args.train_p_ranked_passages, args.train_q_ranked_passages, doc_sampler, worker_id=local_rank, n_workers=args.gpus)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
        val_doc_sampler = SimpleDocumentSampler(args.n_sampled_docs_valid, temperature=args.docs_sampling_temperature, top_k=args.docs_top_k)
        val_dataset = PDataset(args.val_source_path, args.val_target_path, args.val_p_ranked_passages, val_doc_sampler, worker_id=local_rank, n_workers=args.gpus)
        #val_dataset = PQDataset(args.val_source_path, args.val_target_path, args.val_p_ranked_passages, args.val_q_ranked_passages, doc_sampler)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    else:
        assert False, "loss_type not in {NLL, Marginalized, ELBO}"

    logger = CSVLogger(save_dir=args.experiments_directory, name='', version=args.experiment_id)
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', save_top_k=-1,filename="{epoch:02d}-{val_loss:.2f}")
    trainer = Trainer(gpus=args.gpus, logger=logger, default_root_dir=curexpdir, track_grad_norm=2,
                      accumulate_grad_batches=args.accumulate_grad_batches, fast_dev_run=True,
                      callbacks=[checkpoint_callback])
    trainer.fit(model, train_dataloader, val_dataloader)
    trainer = Trainer(gpus=args.gpus, logger=logger,
                      default_root_dir=curexpdir, track_grad_norm=2,
                      accumulate_grad_batches=args.accumulate_grad_batches, accelerator='ddp', callbacks=[checkpoint_callback], max_epochs=args.max_epochs)
    trainer.fit(model, train_dataloader, val_dataloader)


