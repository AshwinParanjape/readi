import argparse
import os
import pickle as pkl
from pathlib import Path
from typing import Dict

import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from meticulous import Experiment
from torch.utils.data.dataloader import default_collate
from transformers import BartForConditionalGeneration, BartTokenizer

from models.qrag import SimpleDocumentSampler, PDataset, MarginalizedLossSystem, Generator, ColBERTScorer, \
    NLLLossSystem, TopKDocumentSampler, InheritableCheckpointMixin, filter_state_dict, collate_fn

class RetrievalScorer(pl.LightningModule):
    def __init__(self, query_maxlen=64, doc_maxlen=256, expdir='', truncate_query_from_start=False, normalize_scorer_embeddings=False, use_scorer='p_scorer', invert_st_order=False):
        super().__init__()
        self.expdir = expdir
        self.p_scorer = ColBERTScorer.from_pretrained('bert-base-uncased',
                                                      truncate_query_from_start = truncate_query_from_start,
                                                      query_maxlen=query_maxlen,
                                                      doc_maxlen=doc_maxlen,
                                                      normalize_embeddings=normalize_scorer_embeddings
                                                      )
        self.q_scorer = ColBERTScorer.from_pretrained('bert-base-uncased',
                                                      truncate_query_from_start = truncate_query_from_start,
                                                      query_maxlen=query_maxlen,
                                                      doc_maxlen=doc_maxlen,
                                                      normalize_embeddings=normalize_scorer_embeddings
                                                      )
        self.instances = []
        self.invert_st_order = invert_st_order
        self.use_scorer=use_scorer


    def test_step(self, batch, batch_idx):
        #print(batch, batch_idx)
        overall_doc_idx = 0
        sources, targets, batched_docs = batch['source'], batch['target'], batch['doc_texts']
        if self.use_scorer == 'p_scorer':
            batched_doc_scores = self.p_scorer(sources, batched_docs)
            
        elif self.use_scorer == 'q_scorer':
            if self.invert_st_order:
                st_text = [t + ' | ' +s for s, t in zip(sources, targets)]
            else:
                st_text = [s + ' | ' +t for s, t in zip(sources, targets)]
            batched_doc_scores = self.q_scorer(st_text, batched_docs)

        for qid, doc_ids, doc_scores, source, target, docs in zip(batch['qid'], batch['doc_ids'], batched_doc_scores, sources, targets, batched_docs):
            instance = {'qid': qid.item(), 'source': source, 'target': target, 'retrievals': []}
            for doc_id, doc, doc_score in sorted(zip(doc_ids, docs, doc_scores), key= lambda tup: -tup[2]):
                doc_gens = {'doc_id': doc_id, 'doc_text': doc, 'doc_score': doc_score.item(), }
                instance['retrievals'].append(doc_gens)
                overall_doc_idx+=1
            self.instances.append(instance)
        return None


def rescore():
    parser = argparse.ArgumentParser(description='Script to rescore documents')
    base_path = Path('/u/scr/ashwinp/research/readi')
    rerank_exp_base_path = Path('/scr/biggest/ashwinp/experiments/colbert-rerank/')
    qtraining_exp_base_path = Path('/scr/biggest/ashwinp/experiments/qtraining/')
    scorer_group = parser.add_argument_group(title='scorer (ColBERT) args')
    scorer_group.add_argument('--query_maxlen', dest='query_maxlen', default=64, type=int)
    scorer_group.add_argument('--doc_maxlen', dest='doc_maxlen', default=184, type=int)
    scorer_group.add_argument('--label_maxlen', dest='label_maxlen', default=64, type=int)
    scorer_group.add_argument('--truncate_query_from_start', action='store_true', default=False)
    scorer_group.add_argument('--unnormalized_scorer_embeddings', action='store_true', default=False)

    paths_group = parser.add_argument_group(title='input file paths')
    paths_group.add_argument('--source_path', type=str, default=(base_path / 'data/wow-kilt/val.source').as_posix(),
                             help='Path to train/val.source file, each line contains input to the generator')
    paths_group.add_argument('--target_path', type=str, default=(base_path / 'data/wow-kilt/val.target').as_posix(),
                             help='Path to train/val.target file, if given will be recorded in output, but not used to generate or compute any values (path is required but optional in spirit)')
    paths_group.add_argument('--ranked_passages', type=str, default=(rerank_exp_base_path / '33/ranking_passages.tsv').as_posix() ,
                             help='Path to ranking_passages.tsv, retrieved and ranked using p-scorer')
    paths_group.add_argument('--scorer_checkpoint', type=str, help="Path to p_scorer checkpoint, can only be qtraining"), #default='/scr/biggest/ashwinp/readi/checkpoints/colbert/colbert-400000.dnn')

    parser.add_argument('--n_sampled_docs', type=int, default=100,
                  help="Number of docs to sample for each instance")
    parser.add_argument('--batch_size', type=int, default=4,
                  help="Number of source strings used at a time")
    parser.add_argument('--limit_batches', type=int, default=1.0, help="Limit number of batches")
    parser.add_argument('--scorer', type=str, default='p_scorer', help='Use a specific scorer (p_scorer, q_scorer)')
    parser.add_argument('--gpus', type=int, default=1, help='number of gpus')

    Experiment.add_argument_group(parser)
    args = parser.parse_args()
    experiment = Experiment.from_parser(parser)
    curexpdir = experiment.curexpdir or './'
    normalize_scorer_embeddings=not args.unnormalized_scorer_embeddings
    model = RetrievalScorer.load_from_checkpoint(args.scorer_checkpoint, strict=False,
            query_maxlen=args.query_maxlen, doc_maxlen=args.doc_maxlen,
            expdir = curexpdir, truncate_query_from_start=args.truncate_query_from_start, 
            normalize_scorer_embeddings = normalize_scorer_embeddings, 
            use_scorer=args.scorer, 
            )

    doc_sampler = TopKDocumentSampler(k=args.n_sampled_docs)
    val_dataset = PDataset(args.source_path, args.target_path, args.ranked_passages, doc_sampler, worker_id=0, n_workers=1)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    trainer = Trainer(gpus=args.gpus, default_root_dir=curexpdir, limit_test_batches=args.limit_batches)
    
    trainer.test(model, test_dataloaders=val_dataloader)
    with open(Path(curexpdir)/'scores.pkl', 'wb') as f:
        pkl.dump(model.instances, f)

if __name__ == '__main__':
    rescore()
