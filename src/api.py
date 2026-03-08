from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np
import os

from src.embeddings import get_model, load_collection, search_similar
from src.clustering import load_clustering_results, get_cluster_for_query
from src.cache import SemanticCache


class QueryRequest(BaseModel):
    query: str
    threshold: float = 0.85

class QueryResponse(BaseModel):
    query: str
    cache_hit: bool
    matched_query: str | None
    similarity_score: float | None
    result: str
    dominant_cluster: int

class CacheStats(BaseModel):
    total_entries: int
    hit_count: int
    miss_count: int
    hit_rate: float
    avg_lookup_ms: float = 0.0

class FlushResponse(BaseModel):
    message: str


state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("loading models...")
    state["model"] = get_model()

    try:
        state["coll"] = load_collection()
        print(f"vector store: {state['coll'].count()} docs")
    except Exception as e:
        print(f"no vector store yet: {e}")
        state["coll"] = None

    try:
        state["clust"] = load_clustering_results()
        nc = state["clust"]["n_clusters"]
        print(f"clustering: {nc} clusters")
    except FileNotFoundError:
        print("no clustering data, run setup.py")
        state["clust"] = None
        nc = 5

    state["cache"] = SemanticCache(thresh=0.85, n_clust=nc)
    print("ready.")
    
    yield
    state.clear()


app = FastAPI(
    title="Trademarkia Semantic Search",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


def do_search(q, q_vec):
    coll = state.get("coll")
    if coll is None:
        return "Vector store not loaded. Run setup.py first."
    
    res = search_similar(q_vec, coll, n=5)
    docs = res["documents"][0]
    if not docs:
        return "No matches."
    
    parts = [f"Top {len(docs)} semantic matches:\n"]
    for i, (doc, dist, meta) in enumerate(zip(docs, res["distances"][0], res["metadatas"][0]), 1):
        sim = 1 - dist
        snippet = doc[:200].replace('\n', ' ')
        parts.append(f"{i}. [{meta['category']}] (similarity: {sim:.3f})\n   {snippet}...")
    return "\n".join(parts)


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(400, "empty query")
    
    model = state["model"]
    cache = state["cache"]
    clust = state.get("clust")

    if req.threshold != cache.thresh:
        cache.set_threshold(req.threshold)

    q_vec = model.encode([req.query], normalize_embeddings=True, convert_to_numpy=True)[0]

    if clust is not None:
        probs, top_c = get_cluster_for_query(q_vec, clust)
    else:
        probs = np.ones(cache.n_clust) / cache.n_clust
        top_c = 0

    hit, matched, sim = cache.lookup(q_vec, probs)
    
    if hit:
        return QueryResponse(
            query=req.query,
            cache_hit=True,
            matched_query=matched.q,
            similarity_score=round(sim, 4),
            result=matched.res,
            dominant_cluster=top_c
        )

    result = do_search(req.query, q_vec)
    cache.store(q=req.query, q_vec=q_vec, res=result, top_c=top_c, probs=probs)

    return QueryResponse(
        query=req.query,
        cache_hit=False,
        matched_query=None,
        similarity_score=round(sim, 4) if sim > -1 else None,
        result=result,
        dominant_cluster=top_c
    )


@app.get("/cache/stats", response_model=CacheStats)
async def cache_stats():
    s = state["cache"].get_stats()
    return CacheStats(
        total_entries=s["total_entries"],
        hit_count=s["hit_count"],
        miss_count=s["miss_count"],
        hit_rate=s["hit_rate"],
        avg_lookup_ms=s.get("avg_lookup_ms", 0.0)
    )

@app.delete("/cache")
async def flush_cache():
    state["cache"].flush()
    return FlushResponse(message="cache flushed")

@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "entries": state["cache"].total_entries
    }
