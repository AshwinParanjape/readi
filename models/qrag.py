import pytorch_lightning as pl
from transformers import BartForConditionalGeneration, BartTokenizer
import sys
sys.path = ['retriever/ColBERT'] + sys.path
from colbert.modeling.tokenization import QueryTokenizer, DocTokenizer

class ColBERT(BertPreTrainedModel):
    def __init__(self, config, query_maxlen, doc_maxlen, mask_punctuation, dim=128, similarity_metric='cosine'):
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
        self.linear = nn.Linear(config.hidden_size, dim, bias=False)

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

        mask = torch.tensor(self.mask(input_ids), device=DEVICE).unsqueeze(2).float()
        D = D * mask

        D = torch.nn.functional.normalize(D, p=2, dim=2)

        if not keep_dims:
            D, mask = D.cpu().to(dtype=torch.float16), mask.cpu().bool().squeeze(-1)
            D = [d[mask[idx]] for idx, d in enumerate(D)]

        return D

    def score(self, Q, D):
        if self.similarity_metric == 'cosine':
            return (Q @ D.permute(0, 2, 1)).max(2).values.sum(1)

        assert self.similarity_metric == 'l2'
        return (-1.0 * ((Q.unsqueeze(2) - D.unsqueeze(1))**2).sum(-1)).max(-1).values.sum(-1)

    def mask(self, input_ids):
        mask = [[(x not in self.skiplist) and (x != 0) for x in d] for d in input_ids.cpu().tolist()]
        return mask

class QRAG(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.generator = BartForConditionalGeneration.from_pretrained("facebook/bart-base", force_bos_token_to_be_generated=True)
        self.generator_tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        self.retriever = ColBERT()
        self.q_retriever = ColBERT()
        self.query_tokenizer = QueryTokenizer(self.retriver.query_maxlen)
        self.doc_tokenizer = DocTokenizer(self.retriever.doc_maxlen)

    def forward(self, context):
        self.quey_tokenizer
        # tokenize context
        # tokenize closed set docs from retriever
        # retriever.score(context, retriever_closed_set)

    def training_step():
        # tokenize context + generation
        # q_retriever.score(context+generation, q_retriever closed set)


    def queryFromText(self, queries, bsize=None, to_cpu=False):
        if bsize:
            batches = self.query_tokenizer.tensorize(queries, bsize=bsize)
            batches = [self.query(input_ids, attention_mask, to_cpu=to_cpu) for input_ids, attention_mask in batches]
            return torch.cat(batches)
        else:
            return self.query_tokenizer.tensorize(queries)
        return self.query(input_ids, attention_mask)

    def docFromText(self, docs, bsize=None, keep_dims=True, to_cpu=False):
        if bsize:
            batches, reverse_indices = self.doc_tokenizer.tensorize(docs, bsize=bsize)

            batches = [self.doc(input_ids, attention_mask, keep_dims=keep_dims, to_cpu=to_cpu)
                       for input_ids, attention_mask in batches]

            if keep_dims:
                D = _stack_3D_tensors(batches)
                return D[reverse_indices]

            D = [d for batch in batches for d in batch]
            return [D[idx] for idx in reverse_indices.tolist()]

        input_ids, attention_mask = self.doc_tokenizer.tensorize(docs)
        return self.doc(input_ids, attention_mask, keep_dims=keep_dims)
