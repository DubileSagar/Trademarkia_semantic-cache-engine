# Trademarkia Neural Search Engine

> Lightweight semantic search over 20 Newsgroups with fuzzy clustering, a first-principles cache layer, and a live FastAPI service — built for the Trademarkia AI/ML Engineer assignment.

---

## Tech Stack

| Component | Choice | Why |
|-----------|--------|-----|
| Embeddings | `all-MiniLM-L6-v2` | 384-dim vectors, best speed/quality ratio for semantic similarity. Smaller than BERT-base (768d) or Ada-002 (1536d) — saves memory and makes cache dot-products fast |
| Vector DB | ChromaDB (local) | No API keys, no external servers. SQLite-backed persistence works perfectly for ~18k docs |
| Clustering | Fuzzy C-Means (`skfuzzy`) | Soft membership — each doc gets a probability distribution across clusters, not a hard label |
| Cache | Custom (no Redis) | Dict bucketed by cluster ID. Lookup is O(n/k) instead of O(n) — the clusters do real architectural work |
| API | FastAPI + uvicorn | Single-command startup, auto-generated Swagger docs |
| Frontend | Vanilla HTML/CSS/JS | Glassmorphism UI with live cache analytics dashboard |

---

## Architecture

```
Query → Embed (MiniLM) → Fuzzy Membership → Cache Lookup
                                                │
                                        Hit ────┤──── Miss
                                         │             │
                                    Return cached   Search ChromaDB
                                                       │
                                                 Store in cache bucket
                                                       │
                                                    Return results
```

---

## Design Decisions

> *"Your design decisions and how you justify them matter as much as the code."*

### Why `all-MiniLM-L6-v2` over larger models?
It's purpose-built for semantic similarity (our exact task). 384 dims instead of 768 or 1536 means the cosine similarity calculations inside the cache are 2-4x faster. For a corpus of 18k newsgroup posts, the quality difference vs `all-mpnet-base-v2` is negligible.

### Why ChromaDB over Pinecone/Milvus/Weaviate?
All three require either cloud accounts or running separate database servers. For 18k documents that's massive overkill. ChromaDB gives us persistent local storage with cosine search built in — zero infrastructure.

### Why PCA before clustering?
Clustering in 384 dimensions hits the curse of dimensionality — all pairwise distances converge, making boundaries meaningless. PCA to 50 dims retains ~90% variance while making FCM actually work.

### How is the number of clusters justified?
Not hardcoded. `setup.py` sweeps k=5 to k=25 and picks the value with the highest **Fuzzy Partition Coefficient (FPC)**. The optimal k is determined programmatically with evidence printed to the console.

### Why is the cache bucketed by cluster?
A flat cache list means every new query checks against *every* stored entry — O(n). By keying on cluster membership, we only compare within the relevant semantic neighborhood — O(n/k). At 10k cached entries with k=5, that's a 5x speedup.

### How does the cache handle rephrased queries?
Cosine similarity on normalized embeddings (dot product). Default threshold: 0.85. "best GPU for gaming" correctly hits "top graphics cards for PC games" (sim ~0.91) but correctly misses "how to program a GPU" (sim ~0.63). The threshold is tunable via the UI slider.

---

## Project Structure

```
├── main.py                 # entry point (uvicorn main:app)
├── setup.py                # one-time: embed corpus, cluster, index
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── src/
│   ├── preprocessing.py    # load + clean 20 newsgroups
│   ├── embeddings.py       # sentence-transformers + chromadb
│   ├── clustering.py       # fuzzy c-means + PCA + boundary analysis
│   ├── cache.py            # cluster-bucketed semantic cache
│   └── api.py              # fastapi endpoints + lifespan
├── static/
│   ├── index.html          # frontend UI
│   ├── app.js              # search + toggle logic
│   └── styles.css          # glassmorphism theme
└── vector_store/           # generated: chromadb + clustering.pkl
```

---

## Quick Start

### Local Setup

```bash
python3 -m venv venv
source venv/bin/activate       # windows: venv\Scripts\activate
pip install -r requirements.txt

python setup.py                # ~5-10 min: embeds corpus, clusters, indexes
uvicorn main:app --port 8000   # start server
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

### Docker

```bash
docker-compose up --build
# server available at http://localhost:8000
```

---

## API Reference

### `POST /query`

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "best graphics card for gaming"}'
```

Response:
```json
{
  "query": "best graphics card for gaming",
  "cache_hit": false,
  "matched_query": null,
  "similarity_score": null,
  "result": "Top 5 semantic matches: ...",
  "dominant_cluster": 3
}
```

On a subsequent similar query:
```json
{
  "query": "top GPU for PC games",
  "cache_hit": true,
  "matched_query": "best graphics card for gaming",
  "similarity_score": 0.9134,
  "result": "Top 5 semantic matches: ...",
  "dominant_cluster": 3
}
```

### `GET /cache/stats`

```json
{
  "total_entries": 42,
  "hit_count": 17,
  "miss_count": 25,
  "hit_rate": 0.405
}
```

### `DELETE /cache`

Flushes all cached entries and resets stats.

---

## Frontend Features

- **Live search** with animated result cards showing matched documents, categories, and similarity scores
- **UI / JSON toggle** on search results — switch between the visual cards and the raw API response JSON
- **UI / JSON toggle** on cache stats sidebar — instantly see the `GET /cache/stats` payload
- **Adjustable threshold slider** — change cache strictness in real-time
- **Flush cache button** — wipe cache and watch stats reset live

---

## Cluster Boundary Analysis

During `setup.py`, the system prints a semantic boundary report:

1. **Core Documents** — high-certainty docs with >85% membership in a single cluster
2. **Boundary Documents** — docs genuinely split between two clusters (e.g., politics + firearms)
3. **Uncertain Documents** — flat distributions where the model has genuine ambiguity

This directly fulfills the requirement: *"Show what lives in them, show what sits at their boundaries, and show where the model is genuinely uncertain."*
