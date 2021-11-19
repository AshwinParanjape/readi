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

from models.qrag import SimpleDocumentSampler, PDataset, Seq2SeqDataset, MarginalizedLossSystem, Generator, FiDGenerator, ColBERTScorer, \
    NLLLossSystem, TopKDocumentSampler, InheritableCheckpointMixin, filter_state_dict


class TargetGenerator(pl.LightningModule, InheritableCheckpointMixin):
    def __init__(self, query_maxlen_generator=64, query_maxlen=64, doc_maxlen=256, label_maxlen=64, expdir='', truncate_query_from_start=False, n_samples_per_doc=8,
            baseline_generator: Generator=None, normalize_scorer_embeddings=False,
            query_sum_topk=None, query_sum_window=None, scorer_agg_fn=torch.sum, FiD=False,
                 **generation_kwargs, ):
        super().__init__()
        self.n_samples_per_doc = n_samples_per_doc
        self.generation_kwargs = generation_kwargs
        self.expdir = expdir
        self._generator = lambda: BartForConditionalGeneration.from_pretrained("facebook/bart-base")#, force_bos_token_to_be_generated=True)
        self._generator_tokenizer_constructor = lambda: BartTokenizer.from_pretrained("facebook/bart-base")
        #self._generator_tokenizer.add_tokens([DOC_TOKEN, TEXT_TOKEN])
        #self.generator = Generator(self._generator, self._generator_tokenizer, truncate_from_start=truncate_query_from_start)
        if FiD:
            self.generator = FiDGenerator(self._generator, self._generator_tokenizer_constructor, input_maxlen=query_maxlen_generator, doc_maxlen=doc_maxlen, output_maxlen=label_maxlen)
        else:
            self.generator = Generator(self._generator, self._generator_tokenizer_constructor, input_maxlen=query_maxlen_generator, doc_maxlen=doc_maxlen, output_maxlen=label_maxlen)
        self.p_scorer = ColBERTScorer.from_pretrained('bert-base-uncased',
                                          truncate_query_from_start = truncate_query_from_start,
                                          query_maxlen=query_maxlen,
                                          doc_maxlen=doc_maxlen,
                                          normalize_embeddings=normalize_scorer_embeddings,
                                          query_sum_topk = query_sum_topk,
                                          query_sum_window = query_sum_window,
                                          agg_fn=scorer_agg_fn,
                                          )
        self.q_scorer = ColBERTScorer.from_pretrained('bert-base-uncased',
                                                      truncate_query_from_start = truncate_query_from_start,
                                                      query_maxlen=query_maxlen,
                                                      doc_maxlen=doc_maxlen,
                                                      normalize_embeddings=normalize_scorer_embeddings,
                                                      query_sum_topk = query_sum_topk,
                                                      query_sum_window = query_sum_window,
                                                      agg_fn=scorer_agg_fn,
                                                      )
        self.FiD = FiD
        self.baseline_generator = baseline_generator
        self.instances = []
        #with open(Path(self.expdir)/ Path('generations.tsv'), 'w') as f:
        #    f.write('q_id\tdoc_id\tp_score\tsource\tgeneration\tpassage\n')

    def set_loss_fn(self):
        pass

    @staticmethod
    def extract_state_dict_from_checkpoints(p_scorer_checkpoint, generator_checkpoint):
        generator_checkpoint = torch.load(generator_checkpoint, torch.device('cpu'))
        state_dict = filter_state_dict(generator_checkpoint, 'generator')
        state_dict.update(filter_state_dict(generator_checkpoint, '_generator'))
        if p_scorer_checkpoint is not None:
            p_scorer_checkpoint = torch.load(p_scorer_checkpoint, torch.device('cpu'))
            state_dict.update(filter_state_dict(p_scorer_checkpoint, 'p_scorer'))
        return state_dict

    def forward(self, batch):
        sources, _ = batch['source'], batch['target']
        try:
            batched_docs = batch['doc_texts']
        except:
            batched_docs = None
        generation_output = self.generator.generate(sources, batched_docs, num_return_sequences=self.n_samples_per_doc,
                                                    **self.generation_kwargs)
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
        sources, targets = batch['source'], batch['target']
        if 'doc_texts' in batch:
            batched_docs = batch['doc_texts']
            batched_doc_scores = self.p_scorer(sources, batched_docs)
        else:
            batched_docs = None
        #doc_log_probs = torch.nn.functional.log_softmax(doc_scores, dim=1)
        generated_output = self(batch)
        if self.baseline_generator is not None:
            generated_output.baseline_log_liklihood = self.baseline_generator.rescore_from_tensors(generated_output.input_encoding, generated_output, self.n_samples_per_doc)
        output_strings = generated_output.strings
        string_idx = 0
        overall_doc_idx = 0
        if batched_docs is None:
            with open(Path(self.expdir)/ Path('generations.tsv'), 'a') as f:
                for qid, source, target in zip(batch['qid'],sources, targets):
                    instance = {'qid': qid.item(), 'source': source, 'target': target}
                    doc_gens = {'generator_output': []}
                    input_ids = generated_output.input_encoding['input_ids'][overall_doc_idx, :]
                    attention_mask = generated_output.input_encoding['attention_mask'][overall_doc_idx, :]
                    sequences = generated_output.sequences[overall_doc_idx*self.n_samples_per_doc:
                                                           (overall_doc_idx+1)*self.n_samples_per_doc, :]
                    strings = generated_output.strings[overall_doc_idx*self.n_samples_per_doc:
                                                           (overall_doc_idx+1)*self.n_samples_per_doc]
                    doc_gens['generator_output'] = {
                        'input_ids': input_ids.tolist(),
                        'attention_mask': attention_mask.tolist(),
                        'sequences': sequences.tolist(),
                        'strings': strings,
                    }

                    overall_doc_idx+=1
                    self.instances.append(instance)
        else:
            with open(Path(self.expdir)/ Path('generations.tsv'), 'a') as f:
                for qid, doc_ids, doc_scores, source, target, docs in zip(batch['qid'], batch['doc_ids'], batched_doc_scores, sources, targets, batched_docs):
                    instance = {'qid': qid.item(), 'source': source, 'target': target, 'retrievals': []}
                    for doc_id, doc, doc_score in zip(doc_ids, docs, doc_scores):
                        doc_gens = {'doc_id': doc_id, 'doc_text': doc, 'doc_score': doc_score.item(), 'generator_output': []}
                        input_ids = generated_output.input_encoding['input_ids'][overall_doc_idx, :]
                        attention_mask = generated_output.input_encoding['attention_mask'][overall_doc_idx, :]
                        sequences = generated_output.sequences[overall_doc_idx*self.n_samples_per_doc:
                                                               (overall_doc_idx+1)*self.n_samples_per_doc, :]
                        strings = generated_output.strings[overall_doc_idx*self.n_samples_per_doc:
                                                               (overall_doc_idx+1)*self.n_samples_per_doc]
                        #log_liklihood = generated_output.log_liklihood[overall_doc_idx*self.n_samples_per_doc:
                        #                                       (overall_doc_idx+1)*self.n_samples_per_doc]
                        #baseline_log_liklihood = generated_output.baseline_log_liklihood[overall_doc_idx*self.n_samples_per_doc:
                        #                                               (overall_doc_idx+1)*self.n_samples_per_doc]
                        doc_gens['generator_output'] = {
                            'input_ids': input_ids.tolist(),
                            'attention_mask': attention_mask.tolist(),
                            'sequences': sequences.tolist(),
                            'strings': strings,
                            #'log_liklihood': log_liklihood.tolist(),
                            #'baseline_log_liklihood': baseline_log_liklihood.tolist(),
                        }

                        instance['retrievals'].append(doc_gens)
                        if not self.FiD:
                            overall_doc_idx+=1
                    if self.FiD:
                        overall_doc_idx+=1
                    self.instances.append(instance)
        return None

def collate_fn(batch: Dict):
    # Differs from the qrag.collate function in that the doc_scores aren't concatenated as tensors

    collated = default_collate(
        [{k:v for k, v in d.items() if k in {'qid', 'source', 'target'}} for d in batch]
    )
    if 'doc_id' in batch[0]:
        collated['doc_ids'] = [d['doc_ids'] for d in batch ]
        collated['doc_texts'] = [d['doc_texts'] for d in batch ]
        collated['doc_scores'] = [d['doc_scores'].tolist() for d in batch ]
    return collated

def generate():
    parser = argparse.ArgumentParser(description='Script to rescore documents and generate samples')
    base_path = Path('/u/scr/ashwinp/research/readi')
    rerank_exp_base_path = Path('/scr/biggest/ashwinp/experiments/colbert-rerank/')
    qtraining_exp_base_path = Path('/scr/biggest/ashwinp/experiments/qtraining/')
    scorer_group = parser.add_argument_group(title='scorer (ColBERT) args')
    scorer_group.add_argument('--query_maxlen_generator', dest='query_maxlen_generator', default=64, type=int)
    scorer_group.add_argument('--query_maxlen', dest='query_maxlen', default=64, type=int)
    scorer_group.add_argument('--doc_maxlen', dest='doc_maxlen', default=184, type=int)
    scorer_group.add_argument('--label_maxlen', dest='label_maxlen', default=64, type=int)
    scorer_group.add_argument('--truncate_query_from_start', action='store_true', default=False)
    scorer_group.add_argument('--unnormalized_scorer_embeddings', action='store_true', default=False)
    scorer_group.add_argument('--query_sum_topk', default=None, type=int)
    scorer_group.add_argument('--query_sum_window', default=None, type=int)
    scorer_group.add_argument('--scorer_agg_fn', default='sum', type=str)

    paths_group = parser.add_argument_group(title='input file paths')
    paths_group.add_argument('--source_path', type=str, default=(base_path / 'data/wow-kilt/val.source').as_posix(),
                             help='Path to train/val.source file, each line contains input to the generator')
    paths_group.add_argument('--target_path', type=str, default=(base_path / 'data/wow-kilt/val.target').as_posix(),
                             help='Path to train/val.target file, if given will be recorded in output, but not used to generate or compute any values (path is required but optional in spirit)')
    paths_group.add_argument('--p_ranked_passages', type=str, default=None,
                             help='Path to ranking_passages.tsv, retrieved and ranked using p-scorer. If not provided, then generates only using source')
    paths_group.add_argument('--p_scorer_checkpoint', type=str, help="Path to p_scorer checkpoint, can only be qtraining"), #default='/scr/biggest/ashwinp/readi/checkpoints/colbert/colbert-400000.dnn')
    paths_group.add_argument('--generator_checkpoint', default=None, type=str, help='Path to generator checkpoint')
    paths_group.add_argument('--no_retrieval_checkpoint', type=str, #default=(qtraining_exp_base_path / '17/checkpoints/epoch=1-step=15763.ckpt').as_posix() ,
                             help='Path to checkpoint which contains the generator and p_scorer model')


    decoding_group = parser.add_argument_group(title='decoding arguments')
    decoding_group.add_argument('--max_length', type=int, default=40,
                                help="The maximum length of the sequence to be generated.")
    decoding_group.add_argument('--min_length', type=int, default=5,
                                help="The minimum length of the sequence to be generated.")
    decoding_group.add_argument('--do_sample',  default=False, action='store_true',
                                help="Whether or not to use sampling ; use greedy decoding otherwise")
    decoding_group.add_argument('--num_beams', type=int, default=1,
                                help="Number of beams for beam search. 1 means no beam search.")
    decoding_group.add_argument('--temperature', type=float, default=1.0, help="The value used to module the next token probabilities.")
    decoding_group.add_argument('--top_k', type=int, default=50, help="The number of highest probability vocabulary tokens to keep for top-k-filtering.")
    decoding_group.add_argument('--top_p', type=float, default=1.0, help="If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation.")

    decoding_group.add_argument('--n_sampled_docs', type=int, default=8,
                                help="Number of docs to sample for each instance")
    decoding_group.add_argument('--n_samples_per_doc', type=int, default=4,
                                help="Number of samples to generate for each doc")
    decoding_group.add_argument('--docs_top_k', type=int, default=100,
                                     help="Sample from top_k docs")
    decoding_group.add_argument('--docs_sampling_temperature', type=float, default=1,
                                     help="Temperature used for sampling docs")
    decoding_group.add_argument('--batch_size', type=int, default=4,
                                help="Number of source strings used at a time")
    decoding_group.add_argument('--limit_batches', type=int, default=1.0, help="Limit number of batches")
    decoding_group.add_argument('--scorer', type=str, default='p_scorer', help='Use a specific scorer (p_scorer, q_scorer)')
    decoding_group.add_argument('--gpus', type=int, default=1, help='number of gpus')
    decoding_group.add_argument('--FiD',  default=False, action='store_true',
                                help="If set, the generator is assumed to use all docs at once")

    Experiment.add_argument_group(parser)
    args = parser.parse_args()
    experiment = Experiment.from_parser(parser)
    #curexpdir = './'
    curexpdir = experiment.curexpdir or './'
    #state_dict = torch.load(args.checkpoint, map_location=torch.device('cpu'))['state_dict']
    #_generator = BartForConditionalGeneration.from_pretrained("facebook/bart-base",
    #                                                               force_bos_token_to_be_generated=True)
    #_generator_tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    #generator = Generator(_generator, _generator_tokenizer)
    #generator.load_state_dict(state_dict={k:v for k, v in state_dict.items() if k.startswith('generator')})
    #p_scorer = ColBERTScorer.load_state_dict(state_dict={k:v for k, v in state_dict.items() if k.startswith('p_scorer')})
    #model = TargetGenerator(generator, p_scorer, expdir=curexpdir, strict=False)


    normalize_scorer_embeddings=not args.unnormalized_scorer_embeddings
    if args.scorer_agg_fn == 'sum':
        scorer_agg_fn = torch.sum
    elif args.scorer_agg_fn == 'mean':
        scorer_agg_fn = torch.mean
    else:
        assert args.scorer_agg_fn in {'sum', 'mean'}, 'Unsupported value of argument scorer_agg_fn'
    state_dict = TargetGenerator.extract_state_dict_from_checkpoints(p_scorer_checkpoint=args.p_scorer_checkpoint,
                                                                   generator_checkpoint=args.generator_checkpoint)
    if args.no_retrieval_checkpoint:
        baseline_model = NLLLossSystem.load_from_checkpoint(args.no_retrieval_checkpoint, strict=False)
    else:
        baseline_model = None
    model = TargetGenerator.init_from_checkpoints(state_dict, expdir=curexpdir,
            query_maxlen_generator=args.query_maxlen_generator,
            query_maxlen=args.query_maxlen, doc_maxlen=args.doc_maxlen, label_maxlen=args.label_maxlen,
            truncate_query_from_start=args.truncate_query_from_start,
            n_samples_per_doc = args.n_samples_per_doc,
            baseline_generator=baseline_model.generator if baseline_model else None,
            top_k = args.top_k,
            top_p=args.top_p,
            temperature = args.temperature,
            do_sample = args.do_sample,
            num_beams = args.num_beams,
            min_length = args.min_length,
            max_length = args.max_length,
            normalize_scorer_embeddings = normalize_scorer_embeddings,
            query_sum_topk=args.query_sum_topk,
            query_sum_window=args.query_sum_window,
            scorer_agg_fn = scorer_agg_fn,
            FiD=args.FiD,
            )
    #doc_sampler = SimpleDocumentSampler(args.n_sampled_docs, temperature=args.docs_sampling_temperature, top_k=args.docs_top_k)
    doc_sampler = TopKDocumentSampler(k=args.n_sampled_docs)
    if args.scorer=='p_scorer':
        val_dataset_scorer = model.p_scorer
    elif args.scorer=='q_scorer':
        val_dataset_scorer = model.q_scorer
    if args.p_ranked_passages is None:
        val_dataset = Seq2SeqDataset(args.source_path, args.target_path, worker_id=0, n_workers=1)
    else:
        val_dataset = PDataset(args.source_path, args.target_path, args.p_ranked_passages, doc_sampler, worker_id=0, n_workers=1,
                               p_scorer=val_dataset_scorer, yield_scores=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn)

    trainer = Trainer(gpus=args.gpus, default_root_dir=curexpdir, limit_test_batches=args.limit_batches)
    trainer.test(model, test_dataloaders=val_dataloader)
    with open(Path(curexpdir)/'generations.pkl', 'wb') as f:
        pkl.dump(model.instances, f)

if __name__ == '__main__':
    generate()

