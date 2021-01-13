import sys, os
sys.path=['../']+sys.path
os.environ['TRANSFORMERS_CACHE'] = "/u/scr/ashwinp/cache/.cache/huggingface/"
os.environ['HF_DATASETS_CACHE']="/u/scr/ashwinp/cache/.cache/huggingface/datasets/"
from models.rag_q import RagTokenizer, RagRetriever, RagSequenceForGeneration

tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq", cache_dir="/u/scr/ashwinp/cache/.cache/huggingface/")
#retriever = RagRetriever.from_pretrained("facebook/rag-sequence-nq", index_name="exact", use_dummy_dataset=True, cache_dir='/u/scr/ashwinp/cache/.cache/huggingface/')
retriever = RagRetriever.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base', dataset="wiki_dpr", index_name='compressed')

model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq", retriever=retriever, cache_dir='/u/scr/ashwinp/cache/.cache/huggingface/')

input_dict = tokenizer.prepare_seq2seq_batch("who is the father of darth vader", return_tensors="pt")
generated = model.generate(input_ids=input_dict["input_ids"])
print(tokenizer.batch_decode(generated, skip_special_tokens=True)[0])

