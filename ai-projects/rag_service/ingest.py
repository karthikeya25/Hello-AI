import os, pickle, faiss, numpy as np
from sentence_transformers import SentenceTransformer
m=SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
texts=[]; paths=[]
for root,_,files in os.walk("docs"):
    for f in files:
        if f.endswith(".txt"):
            p=os.path.join(root,f)
            paths.append(p)
            texts.append(open(p,"r",encoding="utf-8",errors="ignore").read())
emb=m.encode(texts,convert_to_numpy=True,normalize_embeddings=True)
index=faiss.IndexFlatIP(emb.shape[1]); index.add(emb)
faiss.write_index(index,"index.faiss")
pickle.dump({"paths":paths,"texts":texts},open("meta.pkl","wb"))
print(len(texts))
