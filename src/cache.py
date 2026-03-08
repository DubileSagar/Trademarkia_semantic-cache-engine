import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any
import time

DEF_THRESH = 0.85

@dataclass
class CacheEntry:
    q: str
    vec: np.ndarray
    res: Any
    top_c: int
    probs: np.ndarray
    ts: float = field(default_factory=time.time)
    hits: int = 0

class SemanticCache:
    def __init__(self, thresh: float = DEF_THRESH, n_clust: int = 15):
        self.thresh = thresh
        self.n_clust = n_clust
        self.buckets: Dict[int, List[CacheEntry]] = {k: [] for k in range(n_clust)}
        self.hit_ct = 0
        self.miss_ct = 0
        self.lookup_time = 0.0
    
    @staticmethod
    def sim(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))
    
    def lookup(self, q_vec: np.ndarray, probs: np.ndarray) -> Tuple[bool, Optional[CacheEntry], float]:
        t0 = time.time()
        
        targets = [i for i, p in enumerate(probs) if p > 0.01]
        
        best_sim = -1.0
        match = None
        
        for b_id in targets:
            for item in self.buckets[b_id]:
                s = self.sim(q_vec, item.vec)
                if s > best_sim:
                    best_sim = s
                    match = item
        
        self.lookup_time += time.time() - t0
        
        if best_sim >= self.thresh:
            self.hit_ct += 1
            if match:
                match.hits += 1
            return True, match, best_sim
            
        self.miss_ct += 1
        return False, match, best_sim
    
    def store(self, q, q_vec, res, top_c, probs):
        entry = CacheEntry(q=q, vec=q_vec.copy(), res=res, top_c=top_c, probs=probs.copy())
        self.buckets[top_c].append(entry)
        return entry

    def flush(self):
        self.buckets = {k: [] for k in range(self.n_clust)}
        self.hit_ct = 0
        self.miss_ct = 0
        self.lookup_time = 0.0

    @property
    def total_entries(self):
        return sum(len(b) for b in self.buckets.values())
    
    def get_stats(self):
        ops = self.hit_ct + self.miss_ct
        rate = self.hit_ct / ops if ops > 0 else 0.0
        avg_ms = (self.lookup_time / ops * 1000) if ops > 0 else 0.0
        return {
            "total_entries": self.total_entries,
            "hit_count": self.hit_ct,
            "miss_count": self.miss_ct,
            "hit_rate": round(rate, 4),
            "threshold": self.thresh,
            "bucket_sizes": {k: len(v) for k, v in self.buckets.items() if v},
            "avg_lookup_ms": round(avg_ms, 3)
        }
    
    def set_threshold(self, val):
        if not 0.0 <= val <= 1.0:
            raise ValueError("threshold out of range")
        self.thresh = val
