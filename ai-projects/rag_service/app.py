import pickle, faiss, numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
app=FastAPI()
em=SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
tok=AutoTokenizer.from_pretrained("google/flan-t5-base")
gen=AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
index=faiss.read_index("index.faiss")
meta=pickle.load(open("meta.pkl","rb"))
class Q(BaseModel):
    question:str
    k:int=5
def retrieve(q,k):
    v=em.encode([q],normalize_embeddings=True)
    D,I=index.search(v,k)
    ctx=[meta["texts"][i] for i in I[0]]
    return "\n\n".join(ctx)
@app.post("/ask")
def ask(q:Q):
    ctx=retrieve(q.question,q.k)
    prompt=f"Context:\n{ctx}\n\nQuestion:{q.question}\nAnswer concisely:"
    ids=tok(prompt,return_tensors="pt",truncation=True,max_length=1024).input_ids
    out=gen.generate(ids,max_new_tokens=128)
    return {"answer":tok.decode(out[0],skip_special_tokens=True)}
