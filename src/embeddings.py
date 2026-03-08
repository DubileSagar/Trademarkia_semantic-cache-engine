import os
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import chromadb

MODEL_ID = "all-MiniLM-L6-v2"
DB_DIR = "./vector_store"
COLL_NAME = "newsgroups"
BS = 64

def get_model():
    print(f"Loading {MODEL_ID}...")
    return SentenceTransformer(MODEL_ID)

def embed_texts(texts, model, bs=BS):
    print(f"Embedding {len(texts)} docs...")
    vecs = model.encode(
        texts,
        batch_size=bs,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    return vecs

def get_db():
    os.makedirs(DB_DIR, exist_ok=True)
    return chromadb.PersistentClient(path=DB_DIR)

def store_embeddings(texts, vecs, labels, cat_names):
    client = get_db()
    
    try:
        client.delete_collection(COLL_NAME)
        print("Dropped old collection")
    except:
        pass
        
    coll = client.create_collection(name=COLL_NAME, metadata={"hnsw:space": "cosine"})
    
    ids = [f"d_{i}" for i in range(len(texts))]
    meta = [{"label": int(labels[i]), "category": cat_names[int(labels[i])], "doc_id": i} for i in range(len(texts))]
    
    batch_sz = 500
    print("Writing to Chroma...")
    for i in tqdm(range(0, len(texts), batch_sz)):
        end = min(i + batch_sz, len(texts))
        coll.add(
            ids=ids[i:end],
            embeddings=vecs[i:end].tolist(),
            documents=texts[i:end],
            metadatas=meta[i:end]
        )
    return coll

def search_similar(query_vec, coll, n=5):
    return coll.query(
        query_embeddings=[query_vec.tolist()],
        n_results=n,
        include=["documents", "distances", "metadatas"]
    )

def load_collection():
    return get_db().get_collection(COLL_NAME)
