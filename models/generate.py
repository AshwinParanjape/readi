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

from models.qrag import SimpleDocumentSampler, PDataset, collate_fn, MarginalizedLossSystem, Generator, ColBERTScorer, \
    NLLLossSystem, TopKDocumentSampler


class TargetGenerator(pl.LightningModule):
    def __init__(self, query_maxlen=64, doc_maxlen=256, expdir='', truncate_query_from_start=False, n_samples_per_doc=8,
                 baseline_generator: Generator=None,
                 **generation_kwargs):
        super().__init__()
        self.n_samples_per_doc = n_samples_per_doc
        self.generation_kwargs = generation_kwargs
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
        self.baseline_generator = baseline_generator
        self.instances = []
        #with open(Path(self.expdir)/ Path('generations.tsv'), 'w') as f:
        #    f.write('q_id\tdoc_id\tp_score\tsource\tgeneration\tpassage\n')

    def forward(self, batch):
        sources, _, batched_docs = batch['source'], batch['target'], batch['doc_texts']
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
        print(batch, batch_idx)
        sources, targets, batched_docs = batch['source'], batch['target'], batch['doc_texts']
        batched_doc_scores = self.p_scorer(sources, batched_docs)
        #doc_log_probs = torch.nn.functional.log_softmax(doc_scores, dim=1)
        generated_output = self(batch)
        if self.baseline_generator is not None:
            generated_output.baseline_log_liklihood = self.baseline_generator.rescore_from_tensors(generated_output.input_encoding, generated_output, self.n_samples_per_doc)
        output_strings = generated_output.strings
        string_idx = 0
        overall_doc_idx = 0
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
                    log_liklihood = generated_output.log_liklihood[overall_doc_idx*self.n_samples_per_doc:
                                                           (overall_doc_idx+1)*self.n_samples_per_doc]
                    baseline_log_liklihood = generated_output.baseline_log_liklihood[overall_doc_idx*self.n_samples_per_doc:
                                                                   (overall_doc_idx+1)*self.n_samples_per_doc]
                    doc_gens['generator_output'] = {
                        'input_ids': input_ids.tolist(),
                        'attention_mask': attention_mask.tolist(),
                        'sequences': sequences.tolist(),
                        'strings': strings,
                        'log_liklihood': log_liklihood.tolist(),
                        'baseline_log_liklihood': baseline_log_liklihood.tolist(),
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
    scorer_group.add_argument('--query_maxlen', dest='query_maxlen', default=64, type=int)
    scorer_group.add_argument('--doc_maxlen', dest='doc_maxlen', default=180, type=int)
    scorer_group.add_argument('--truncate_query_from_start', action='store_true', default=False)

    paths_group = parser.add_argument_group(title='input file paths')
    paths_group.add_argument('--source_path', type=str, default=(base_path / 'data/wow-kilt/val.source').as_posix(),
                             help='Path to train/val.source file, each line contains input to the generator')
    paths_group.add_argument('--target_path', type=str, default=(base_path / 'data/wow-kilt/val.target').as_posix(),
                             help='(Optional) Path to train/val.target file, each line contains input to the generator')
    paths_group.add_argument('--p_ranked_passages', type=str, default=(rerank_exp_base_path / '33/ranking_passages.tsv').as_posix() ,
                             help='Path to ranking_passages.tsv, retrieved and ranked using p-scorer')
    paths_group.add_argument('--checkpoint', type=str, default=(qtraining_exp_base_path / '16/checkpoints/epoch=2-step=11822.ckpt').as_posix() ,
                             help='Path to checkpoint which contains the generator and p_scorer model')
    paths_group.add_argument('--no_retrieval_checkpoint', type=str, default=(qtraining_exp_base_path / '17/checkpoints/epoch=1-step=15763.ckpt').as_posix() ,
                             help='Path to checkpoint which contains the generator and p_scorer model')


    decoding_group = parser.add_argument_group(title='decoding arguments')
    decoding_group.add_argument('--max_length', type=int, default=40,
                                help="The maximum length of the sequence to be generated.")
    decoding_group.add_argument('--min_length', type=int, default=5,
                                help="The minimum length of the sequence to be generated.")
    decoding_group.add_argument('--do_sample', type=bool, default=True,
                                help="Whether or not to use sampling ; use greedy decoding otherwise")
    decoding_group.add_argument('--num_beams', type=bool, default=8,
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
    baseline_model = NLLLossSystem.load_from_checkpoint(args.no_retrieval_checkpoint, strict=False)
    model = TargetGenerator.load_from_checkpoint(args.checkpoint, strict=False, expdir=curexpdir,
                                                 query_maxlen=args.query_maxlen, doc_maxlen=args.doc_maxlen,
                                                 truncate_query_from_start=args.truncate_query_from_start,
                                                 n_samples_per_doc = args.n_samples_per_doc,
                                                 baseline_generator=baseline_model.generator,
                                                 top_k = args.top_k,
                                                 top_p=args.top_p,
                                                 temperature = args.temperature,
                                                 do_sample = args.do_sample,
                                                 num_beams = args.num_beams,
                                                 min_length = args.min_length,
                                                 max_length = args.max_length,
                                                 )
    #doc_sampler = SimpleDocumentSampler(args.n_sampled_docs, temperature=args.docs_sampling_temperature, top_k=args.docs_top_k)
    doc_sampler = TopKDocumentSampler(k=args.n_sampled_docs)
    val_dataset = PDataset(args.source_path, args.target_path, args.p_ranked_passages, doc_sampler, worker_id=0, n_workers=1,
                           p_scorer=model.p_scorer)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn)

    trainer = Trainer(gpus=1, default_root_dir=curexpdir, limit_test_batches=args.limit_batches)
    trainer.test(model, test_dataloaders=val_dataloader)
    with open(Path(curexpdir)/'generations.pkl', 'wb') as f:
        pkl.dump(model.instances, f)

if __name__ == '__main__':
    generate()

