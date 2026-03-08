import numpy as np
import pickle
import os
from sklearn.decomposition import PCA
import skfuzzy as fuzz

N_COMP = 50
M = 2.0
ERR = 0.005
MAX_IT = 200
CACHE_PATH = "./vector_store/clustering.pkl"


def reduce_dimensions(vecs, n=N_COMP):
    print(f"PCA: {vecs.shape[1]}D -> {n}D")
    pca = PCA(n_components=n, random_state=42)
    out = pca.fit_transform(vecs)
    print(f"Variance kept: {pca.explained_variance_ratio_.sum():.1%}")
    return out, pca


def find_optimal_clusters(data_reduced, k_range=range(5, 26, 5)):
    print("\nSearching optimal k via FPC...")
    best_fpc, best_k, best = -1.0, 15, None
    X = data_reduced.T

    for k in k_range:
        print(f"  k={k}...", end=" ", flush=True)
        cntr, u, _, _, _, _, fpc = fuzz.cluster.cmeans(
            X, c=k, m=M, error=ERR, maxiter=MAX_IT, init=None, seed=42
        )
        print(f"FPC={fpc:.4f}")
        if fpc > best_fpc:
            best_fpc = fpc
            best_k = k
            best = (cntr, u, fpc)

    print(f"\n=> optimal k={best_k} (FPC={best_fpc:.4f})")
    return best_k, best


def get_document_clusters(u):
    probs = u.T
    return np.argmax(probs, axis=1), probs


def save_clustering_results(cntr, u, pca):
    os.makedirs("./vector_store", exist_ok=True)
    blob = {
        "centroids": cntr, "memberships": u,
        "pca_model": pca, "n_clusters": cntr.shape[0],
        "n_docs": u.shape[1]
    }
    with open(CACHE_PATH, 'wb') as f:
        pickle.dump(blob, f)
    print(f"saved clustering to {CACHE_PATH}")


def load_clustering_results():
    if not os.path.exists(CACHE_PATH):
        raise FileNotFoundError("run setup.py first")
    with open(CACHE_PATH, 'rb') as f:
        return pickle.load(f)


def get_cluster_for_query(q_vec, clust_data):
    pca = clust_data["pca_model"]
    centers = clust_data["centroids"]

    q_pca = pca.transform(q_vec.reshape(1, -1))
    dists = np.linalg.norm(centers - q_pca, axis=1)

    exp = 2.0 / (M - 1)

    if np.any(dists == 0):
        probs = np.zeros(len(dists))
        probs[np.argmin(dists)] = 1.0
        return probs, int(np.argmin(dists))

    probs = np.zeros(len(dists))
    for i in range(len(dists)):
        probs[i] = 1.0 / np.sum((dists[i] / dists) ** exp)
    return probs, int(np.argmax(probs))


def analyze_clusters(texts, top_c, probs, labels, cat_names):
    nc = probs.shape[1]
    print("\n" + "="*60)
    print("CLUSTER BOUNDARY ANALYSIS")
    print("="*60)

    # core docs (high certainty)
    print("\n1. Core documents (>85% membership)")
    shown = 0
    for c in range(nc):
        col = probs[:, c]
        idx = np.argmax(col)
        if col[idx] > 0.85:
            snip = texts[idx][:80].replace('\n', ' ')
            print(f"  cluster {c}: prob={col[idx]:.2f} | {snip}... [{cat_names[labels[idx]]}]")
            shown += 1
            if shown >= 3:
                break

    # boundary docs (ambiguous)
    print("\n2. Boundary documents (split membership)")
    sorted_probs = np.sort(probs, axis=1)
    gaps = sorted_probs[:, -1] - sorted_probs[:, -2]
    for idx in np.argsort(gaps)[:3]:
        row = probs[idx]
        top2 = np.argsort(row)[::-1][:2]
        snip = texts[idx][:80].replace('\n', ' ')
        print(f"  c{top2[0]}={row[top2[0]]:.2f} | c{top2[1]}={row[top2[1]]:.2f}")
        print(f"    {snip}... [{cat_names[labels[idx]]}]\n")

    # noisy docs (flat distribution)
    print("3. Uncertain documents (high entropy)")
    ent = -np.sum(probs * np.log(probs + 1e-10), axis=1)
    for idx in np.argsort(ent)[::-1][:2]:
        tc = np.argmax(probs[idx])
        snip = texts[idx][:80].replace('\n', ' ')
        print(f"  max_prob={probs[idx][tc]:.2f} | {snip}... [{cat_names[labels[idx]]}]")

    print("="*60 + "\n")
