"""
HDC LSH Layer V2
Recherche approximative de plus proche voisin en temps constant amorti.
Utilise plusieurs tables de hachage (L) avec des hyperplans binaire (K).
"""

import numpy as np
from typing import Callable, Iterable

class LSHIndex:
    def __init__(self, dim: int, num_tables: int = 10, num_bits: int = 12):
        self.dim = dim
        self.L = num_tables
        self.K = num_bits
        
        # Hyperplans aléatoires pour chaque table
        # Une table i utilise K vecteurs binaires de dimension dim
        self.planes = [
            np.random.randint(0, 2, size=(self.K, self.dim), dtype=np.uint8)
            for _ in range(self.L)
        ]
        
        # Tables de hachage : list[dict[hash_key, list[item_index]]]
        self.tables = [{} for _ in range(self.L)]
        self.items = [] # Liste des tokens ou contextes stockés
        self.hvs = None # Matrice des hypervecteurs correspondants
        
    def _hash(self, hv: np.ndarray, table_idx: int) -> int:
        """
        Calcule une empreinte de K bits.
        bit j = 1 si hamming(hv, planes[j]) < dim/2
        """
        planes = self.planes[table_idx]
        # XOR + count_nonzero sur chaque ligne
        # (K, dim) XOR (dim,) -> (K, dim)
        diffs = np.bitwise_xor(planes, hv[np.newaxis, :])
        counts = np.count_nonzero(diffs, axis=1)
        
        # Empreinte binaire : 1 si distance < dim/2
        bits = (counts < self.dim // 2).astype(np.uint8)
        
        # Conversion en entier (hash key)
        # On utilise packbits ou une simple boucle
        return hash(bits.tobytes())

    def build(self, vocabulary: list[str], encoder: Callable[[str], np.ndarray]):
        """Construit l'index pour un vocabulaire donné."""
        self.items = vocabulary
        self.hvs = np.stack([encoder(t) for t in vocabulary])
        
        for i, hv in enumerate(self.hvs):
            for l in range(self.L):
                h = self._hash(hv, l)
                if h not in self.tables[l]:
                    self.tables[l][h] = []
                self.tables[l][h].append(i)

    def query(self, query_hv: np.ndarray, k: int = 5) -> list[tuple[str, int]]:
        """Recherche les k plus proches voisins."""
        candidates = set()
        for l in range(self.L):
            h = self._hash(query_hv, l)
            if h in self.tables[l]:
                candidates.update(self.tables[l][h])
        
        if not candidates:
            return []
            
        # Scan exact uniquement sur les candidats
        cand_indices = list(candidates)
        cand_hvs = self.hvs[cand_indices]
        
        diffs = np.bitwise_xor(cand_hvs, query_hv[np.newaxis, :])
        distances = np.count_nonzero(diffs, axis=1)
        
        # Top-k
        top_k_rel_idx = np.argsort(distances)[:k]
        return [
            (self.items[cand_indices[i]], int(distances[i]))
            for i in top_k_rel_idx
        ]

    @property
    def size(self) -> int:
        return len(self.items)
