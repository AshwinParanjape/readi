import argparse
import os
import pickle as pkl
from pathlib import Path

import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from meticulous import Experiment
from transformers import BartForConditionalGeneration, BartTokenizer

from models.qrag import SimpleDocumentSampler, PDataset, collate_fn, MarginalizedLossSystem, Generator, ColBERTScorer


class TargetGenerator(pl.LightningModule):
    def __init__(self, query_maxlen=64, doc_maxlen=256, expdir='', do_sample=True, num_return_sequences_per_doc=8, truncate_query_from_start=False):
        super().__init__()
        self.do_sample = do_sample
        self.num_return_sequences_per_doc = num_return_sequences_per_doc
        self.expdir = expdir
        self._generator = BartForConditionalGeneration.from_pretrained("facebook/bart-base")#, force_bos_token_to_be_generated=True)
        self._generator_tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        #self._generator_tokenizer.add_tokens([DOC_TOKEN, TEXT_TOKEN])
        #self.generator = Generator(self._generator, self._generator_tokenizer, truncate_from_start=truncate_query_from_start)
        self.generator = Generator(self._generator, self._generator_tokenizer)
        self.p_scorer = ColBERTScorer.from_pretrained('bert-base-uncased',
                                          truncate_query_from_start = truncate_query_from_start,
                                          query_maxlen=query_maxlen,
                                          doc_maxlen=doc_maxlen,
                                          )
        self.instances = []
        #with open(Path(self.expdir)/ Path('generations.tsv'), 'w') as f:
        #    f.write('q_id\tdoc_id\tp_score\tsource\tgeneration\tpassage\n')

    def forward(self, batch):
        sources, _, batched_docs = batch['source'], batch['target'], batch['doc_texts']
        generation_output = self.generator.generate(sources, batched_docs, num_return_sequences=self.num_return_sequences_per_doc)
        #batched_doc_scores = self.p_scorer(sources, batched_docs)
        #doc_log_probs = torch.nn.functional.log_softmax(doc_scores, dim=1)
        #output_strings = generation_output.strings
        #string_idx = 0
        #with open(Path(self.expdir)/ Path('generations.tsv'), 'a') as f:
        #    for qid, doc_ids, doc_scores, source, docs in enumerate(zip(batch['qids'], batch['doc_ids'], batched_doc_scores, sources, batched_docs)):
        #        for doc_id, doc, doc_score in enumerate(doc_ids, docs, doc_scores):
        #            for gen_id in range(self.num_return_sequences_per_doc):
        #                f.write(f'{qid}\t{doc_id}\t{doc_score}\t{source}\t{output_strings[string_idx]}\t{doc}\n')
        #                string_idx+=1
        return generation_output

    def test_step(self, batch, batch_idx):
        print(batch, batch_idx)
        sources, _, batched_docs = batch['source'], batch['target'], batch['doc_texts']
        batched_doc_scores = self.p_scorer(sources, batched_docs)
        #doc_log_probs = torch.nn.functional.log_softmax(doc_scores, dim=1)
        generated_output = self(batch)
        output_strings = generated_output.strings
        string_idx = 0
        overall_doc_idx = 0
        with open(Path(self.expdir)/ Path('generations.tsv'), 'a') as f:
            for qid, doc_ids, doc_scores, source, docs in zip(batch['qid'], batch['doc_ids'], batched_doc_scores, sources, batched_docs):
                instance = {'qid': qid, 'source': source, 'retrievals': []}
                for doc_id, doc, doc_score in zip(doc_ids, docs, doc_scores):
                    doc_gens = {'doc_id': doc_id, 'doc_text': doc, 'doc_score': doc_score, 'generator_output': []}
                    input_ids = generated_output.input_encoding['input_ids'][overall_doc_idx, :]
                    attention_mask = generated_output.input_encoding['attention_mask'][overall_doc_idx, :]
                    sequences = generated_output.sequences[overall_doc_idx*self.num_return_sequences_per_doc:
                                                           (overall_doc_idx+1)*self.num_return_sequences_per_doc, :]
                    strings = generated_output.strings[overall_doc_idx*self.num_return_sequences_per_doc:
                                                           (overall_doc_idx+1)*self.num_return_sequences_per_doc]
                    log_liklihood = generated_output.log_liklihood[overall_doc_idx*self.num_return_sequences_per_doc:
                                                           (overall_doc_idx+1)*self.num_return_sequences_per_doc]
                    doc_gens['generator_output'] = {
                        'input_ids': input_ids,
                        'attention_mask': attention_mask,
                        'sequences': sequences,
                        'strings': strings,
                        'log_liklihood': log_liklihood
                    }

                    instance['retrievals'].append(doc_gens)
                    overall_doc_idx+=1
            self.instances.append(instance)
        return generated_output


def generate():
    parser = argparse.ArgumentParser(description='Script to jointly train retriever and generator')
    base_path = Path('/u/scr/ashwinp/research/readi')
    rerank_exp_base_path = Path('/scr/biggest/ashwinp/experiments/colbert-rerank/')
    qtraining_exp_base_path = Path('/scr/biggest/ashwinp/experiments/qtraining/')
    scorer_group = parser.add_argument_group(title='scorer (ColBERT) args')
    scorer_group.add_argument('--p_scorer_checkpoint', default='/scr/biggest/ashwinp/readi/checkpoints/colbert/colbert-400000.dnn')
    scorer_group.add_argument('--query_maxlen', dest='query_maxlen', default=64, type=int)
    scorer_group.add_argument('--doc_maxlen', dest='doc_maxlen', default=180, type=int)
    scorer_group.add_argument('--truncate_query_from_start', action='store_true', default=False)

    paths_group = parser.add_argument_group(title='input file paths')
    paths_group.add_argument('--source_path', type=str, default=(base_path / 'data/wow-kilt/val.source').as_posix(),
                             help='Path to train.source file, each line contains input to the generator')
    paths_group.add_argument('--p_ranked_passages', type=str, default=(rerank_exp_base_path / '33/ranking_passages.tsv').as_posix() ,
                             help='Path to ranking_passages.tsv, retrieved and ranked using p-scorer')
    paths_group.add_argument('--checkpoint', type=str, default=(qtraining_exp_base_path / '16/checkpoints/epoch=2-step=11822.ckpt').as_posix() ,
                             help='Path to checkpoint which contains the generator and p_scorer model')


    decoding_group = parser.add_argument_group(title='decoding arguments')
    decoding_group.add_argument('--n_sampled_docs', type=int, default=8,
                                     help="Number of docs to sample for each instance (Marginalized, ELBO)")
    decoding_group.add_argument('--docs_top_k', type=int, default=100,
                                     help="Sample from top_k docs (Marginalized, ELBO)")
    decoding_group.add_argument('--docs_sampling_temperature', type=float, default=1,
                                     help="Temperature used for sampling docs (Marginalized, ELBO)")
    decoding_group.add_argument('--batch_size', type=int, default=8,
                                help="Number of source strings used at a time")

    #Experiment.add_argument_group(parser)
    args = parser.parse_args()
    #experiment = Experiment.from_parser(parser)
    curexpdir = './'
    #curexpdir = experiment.curexpdir
    #state_dict = torch.load(args.checkpoint, map_location=torch.device('cpu'))['state_dict']
    #_generator = BartForConditionalGeneration.from_pretrained("facebook/bart-base",
    #                                                               force_bos_token_to_be_generated=True)
    #_generator_tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    #generator = Generator(_generator, _generator_tokenizer)
    #generator.load_state_dict(state_dict={k:v for k, v in state_dict.items() if k.startswith('generator')})
    #p_scorer = ColBERTScorer.load_state_dict(state_dict={k:v for k, v in state_dict.items() if k.startswith('p_scorer')})
    #model = TargetGenerator(generator, p_scorer, expdir=curexpdir, strict=False)
    model = TargetGenerator.load_from_checkpoint(args.checkpoint, strict=False, query_maxlen=args.query_maxlen, doc_maxlen=args.doc_maxlen)
    doc_sampler = SimpleDocumentSampler(args.n_sampled_docs, temperature=args.docs_sampling_temperature, top_k=args.docs_top_k)
    val_dataset = PDataset(args.source_path, None, args.p_ranked_passages, doc_sampler, worker_id=0, n_workers=1)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn)

    trainer = Trainer(gpus=1, default_root_dir=curexpdir)
    trainer.test(model, test_dataloaders=val_dataloader)
    with open(Path(curexpdir)/'generations.pkl', 'wb'):
        pkl.dump(model.instances)

if __name__ == '__main__':
    generate()

