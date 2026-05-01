"""
HDC LSH Layer V2
Recherche approximative de plus proche voisin en temps constant amorti.
"""

import numpy as np
from typing import Callable, Iterable

class LSHIndex:
    def __init__(self, dim: int, num_tables: int = 10, num_bits: int = 12):
        self.dim = dim
        self.L = num_tables
        self.K = num_bits
        
        self.planes = [
            np.random.randint(0, 2, size=(self.K, self.dim), dtype=np.uint8)
            for _ in range(self.L)
        ]
        
        self.tables = [{} for _ in range(self.L)]
        self.items = [] 
        self.hvs = None 
        
    def _hash(self, hv: np.ndarray, table_idx: int) -> int:
        planes = self.planes[table_idx]
        diffs = np.bitwise_xor(planes, hv[np.newaxis, :])
        counts = np.count_nonzero(diffs, axis=1)
        bits = (counts < self.dim // 2).astype(np.uint8)
        return hash(bits.tobytes())

    def build(self, vocabulary: list[str], encoder: Callable[[str], np.ndarray]):
        """Construit l'index pour un vocabulaire donné."""
        self.items = vocabulary
        self.hvs = np.stack([encoder(t) for t in vocabulary])
        self.build_precomputed(vocabulary, self.hvs)

    def build_precomputed(self, items: list[str], hvs: np.ndarray):
        """Version optimisée si les HVs sont déjà calculés."""
        self.items = items
        self.hvs = hvs
        for i, hv in enumerate(self.hvs):
            for l in range(self.L):
                h = self._hash(hv, l)
                if h not in self.tables[l]:
                    self.tables[l][h] = []
                self.tables[l][h].append(i)

    def query_indices(self, query_hv: np.ndarray, k: int = 20) -> list[int]:
        """Retourne uniquement les indices des candidats (filtrage rapide)."""
        candidates = set()
        for l in range(self.L):
            h = self._hash(query_hv, l)
            if h in self.tables[l]:
                candidates.update(self.tables[l][h])
        return list(candidates)

    def query(self, query_hv: np.ndarray, k: int = 5) -> list[tuple[str, int]]:
        """Recherche les k plus proches voisins."""
        cand_indices = self.query_indices(query_hv, k=k*4)
        if not cand_indices:
            return []
            
        cand_hvs = self.hvs[cand_indices]
        diffs = np.bitwise_xor(cand_hvs, query_hv[np.newaxis, :])
        distances = np.count_nonzero(diffs, axis=1)
        
        top_k_rel_idx = np.argsort(distances)[:k]
        return [
            (self.items[cand_indices[i]], int(distances[i]))
            for i in top_k_rel_idx
        ]

    @property
    def size(self) -> int:
        return len(self.items)
