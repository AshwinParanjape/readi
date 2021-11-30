import pytorch_lightning as pl
class
class MarginalizedLossSystem(pl.LightningModule, InheritableCheckpointMixin):
    def __init__(self, p_scorer_constructor, generator_constructor, expdir='', lr=1e-3, fix_scorer=False):
        super().__init__()
        self.generator = generator_constructor()
        self.p_scorer = p_scorer_constructor()
        self.fix_scorer = fix_scorer
        if self.fix_scorer:
            self.p_scorer.eval()
            for param in self.p_scorer.parameters():
                param.requires_grad = False
        # saved_state_dict = torch.load(p_scorer_checkpoint, map_location='cpu')
        # self.p_scorer.load_state_dict(saved_state_dict['model_state_dict'], strict=False)
        self.set_loss_fn()
        print(self.parameters())

        self.expdir = expdir
        self.lr = lr

    @staticmethod
    def extract_state_dict_from_colbert_checkpoints(p_scorer_checkpoint):
        p_scorer_checkpoint = torch.load(p_scorer_checkpoint, torch.device('cpu'))
        state_dict = {'p_scorer.' + k: v for k, v in p_scorer_checkpoint.items()}
        return state_dict

    def set_loss_fn(self):
        if self.fix_scorer:
            self.p_scorer.eval()
            print("Fixed p scorer")
        self.loss_fn = MarginalizedNLLFn(self.p_scorer, self.generator, self.fix_scorer)

    @staticmethod
    def extract_state_dict_from_checkpoints(p_scorer_checkpoint, generator_checkpoint):
        state_dict = {}
        p_scorer_checkpoint = torch.load(p_scorer_checkpoint, torch.device('cpu'))
        state_dict.update(filter_state_dict(p_scorer_checkpoint, 'p_scorer'))
        if generator_checkpoint is not None:
            generator_checkpoint = torch.load(generator_checkpoint, torch.device('cpu'))
            state_dict.update(filter_state_dict(generator_checkpoint, 'generator'))
            state_dict.update(filter_state_dict(generator_checkpoint, '_generator'))
        return state_dict

    def training_step(self, batch, batch_idx):
        # ['qid': List[int], 'source':List[str], 'target':List[str], 'doc_ids': List[List[int]], 'doc_texts': List[List[str]]]
        if self.fix_scorer: self.p_scorer.eval()
        output: MarginalizedNLL = self.loss_fn(batch['source'], batch['target'], batch['doc_texts'])

        log_value(Path(self.expdir) / Path('metrics.tsv'), 'train', self.current_epoch, batch_idx, 'loss',
                  output.loss)

        log_batch_value(Path(self.expdir) / Path('p_scores.tsv'), 'train', self.current_epoch, batch['qid'],
                        batch['doc_ids'], output.p_scores)
        log_batch_value(Path(self.expdir) / Path('nll.tsv'), 'train', self.current_epoch, batch['qid'],
                        batch['doc_ids'], output.lm_nll)

        return output.loss

    def validation_step(self, batch, batch_idx):
        # ['qid': List[int], 'source':List[str], 'target':List[str], 'doc_ids': List[List[int]], 'doc_texts': List[List[str]]]
        output: MarginalizedNLL = self.loss_fn(batch['source'], batch['target'], batch['doc_texts'])
        output: MarginalizedNLL = self.loss_fn(batch['source'], batch['target'], batch['doc_texts'])

        log_value(Path(self.expdir) / Path('metrics.tsv'), 'val', self.current_epoch, batch_idx, 'loss',
                  output.loss)

        log_batch_value(Path(self.expdir) / Path('p_scores.tsv'), 'val', self.current_epoch, batch['qid'],
                        batch['doc_ids'], output.p_scores)
        log_batch_value(Path(self.expdir) / Path('nll.tsv'), 'val', self.current_epoch, batch['qid'],
                        batch['doc_ids'], output.lm_nll)
        return output.loss

    def configure_optimizers(self):
        if self.fix_scorer:
            optimizer = torch.optim.Adam(itertools.chain(self.generator.parameters()), lr=self.lr)
        else:
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
