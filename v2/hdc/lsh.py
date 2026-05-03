"""
HDC LSH Layer V4 (Bit-Packed)
Optimise pour travailler sur des vecteurs compresse (np.packbits).
"""
import numpy as np
from typing import Callable

class LSHIndex:
    def __init__(self, dim: int, num_tables: int = 10, num_bits: int = 12):
        self.dim    = dim
        self.L      = num_tables
        self.K      = num_bits
        self.half_D = dim // 2

        # Hyperplans fixes (STOCKAGE PACKE)
        rng = np.random.default_rng(seed=1234)
        planes_raw = rng.integers(0, 2, size=(self.L, self.K, self.dim), dtype=np.uint8)
        self._planes_packed = np.packbits(planes_raw, axis=-1)

        self.tables: list[dict[int, list[int]]] = [{} for _ in range(self.L)]
        self.items:  list[str]       = []
        self.hvs_packed: np.ndarray | None = None
        self._key_cache: np.ndarray | None = None

    def _hash_batch(self, hvs_packed: np.ndarray) -> np.ndarray:
        """Calcule les cles LSH pour un batch de HVs packes."""
        V = len(hvs_packed)
        all_keys = np.zeros((V, self.L), dtype=np.uint16)
        
        # On doit deballer pour le XOR bit-a-bit (Numpy ne supporte pas XOR + popcount natif sur uint8 packe)
        # Mais on le fait par petits chunks pour economiser la RAM
        chunk_size = 500
        for start in range(0, V, chunk_size):
            end = min(start + chunk_size, V)
            batch_bits = np.unpackbits(hvs_packed[start:end], axis=-1)[:, :self.dim]
            
            for t in range(self.L):
                planes_bits = np.unpackbits(self._planes_packed[t], axis=-1)[:, :self.dim]
                # Hamming : (C, 1, D) != (1, K, D) -> (C, K, D) -> sum -> (C, K)
                dists = (batch_bits[:, None, :] != planes_bits[None, :, :]).sum(axis=-1)
                bits  = (dists < self.half_D).astype(np.uint8)
                packed = np.packbits(bits, axis=-1)
                
                # Conversion en uint16
                if packed.shape[1] >= 2:
                    keys_t = packed[:, 0].astype(np.uint16) * 256 + packed[:, 1]
                else:
                    keys_t = packed[:, 0].astype(np.uint16)
                all_keys[start:end, t] = keys_t
        return all_keys

    def _hash_one(self, hv_packed: np.ndarray) -> list[int]:
        """Hash un seul vecteur packe."""
        hv_bits = np.unpackbits(hv_packed)[:self.dim]
        keys = []
        for t in range(self.L):
            planes_bits = np.unpackbits(self._planes_packed[t], axis=-1)[:, :self.dim]
            dists = (hv_bits != planes_bits).sum(axis=-1)
            bits  = (dists < self.half_D).astype(np.uint8)
            packed = np.packbits(bits)
            k = int(packed[0]) * 256 + int(packed[1]) if len(packed) >= 2 else int(packed[0])
            keys.append(k)
        return keys

    def build_precomputed(self, items: list[str], hvs_packed: np.ndarray):
        self.items = items
        self.hvs_packed = hvs_packed
        
        if self._key_cache is None or len(self._key_cache) != len(items):
            self._key_cache = self._hash_batch(hvs_packed)
            
        self.tables = [{} for _ in range(self.L)]
        for i in range(len(items)):
            for t in range(self.L):
                k = int(self._key_cache[i, t])
                if k not in self.tables[t]: self.tables[t][k] = []
                self.tables[t][k].append(i)

    def query(self, query_hv_packed: np.ndarray, k: int = 5) -> list[tuple[str, int]]:
        keys = self._hash_one(query_hv_packed)
        candidates = set()
        for t, key in enumerate(keys):
            if key in self.tables[t]: candidates.update(self.tables[t][key])
        
        if not candidates: return []
        
        cand_idx = list(candidates)
        cand_hvs_packed = self.hvs_packed[cand_idx]
        
        # Hamming sur packed
        query_bits = np.unpackbits(query_hv_packed)[:self.dim]
        cand_bits  = np.unpackbits(cand_hvs_packed, axis=-1)[:, :self.dim]
        dists = (query_bits != cand_bits).sum(axis=-1)
        
        top_rel = np.argsort(dists)[:k]
        return [(self.items[cand_idx[i]], int(dists[i])) for i in top_rel]
