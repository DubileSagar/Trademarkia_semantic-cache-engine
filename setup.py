import numpy as np
import os

from src.preprocessing import prepare_corpus
from src.embeddings import get_model, embed_texts, store_embeddings
from src.clustering import (
    reduce_dimensions, get_document_clusters,
    save_clustering_results, analyze_clusters, find_optimal_clusters
)

def main():
    print("Setting up search engine...\n")

    texts, labels, cats = prepare_corpus()

    print("\nGenerating embeddings...")
    model = get_model()
    vecs = embed_texts(texts, model)

    os.makedirs("./vector_store", exist_ok=True)
    np.save("./vector_store/embeddings.npy", vecs)

    print("\nIndexing into ChromaDB...")
    store_embeddings(texts, vecs, labels, cats)

    print("\nClustering...")
    reduced, pca = reduce_dimensions(vecs)
    best_k, (cntr, u, fpc) = find_optimal_clusters(reduced, k_range=range(5, 26, 5))
    top_c, probs = get_document_clusters(u)
    save_clustering_results(cntr, u, pca)

    print("\nAnalyzing boundary docs...")
    analyze_clusters(texts, top_c, probs, labels, cats)

    print("\nDone. Run: uvicorn main:app --port 8000")

if __name__ == "__main__":
    main()
