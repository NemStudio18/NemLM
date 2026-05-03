"""
HDC Associative Memory Layer V3 (Persistent & Bit-Packed)
Support pour le stockage sur disque via BTree (SQLite) pour gérer des millions de contextes sur 16GB.
"""
import numpy as np
import sqlite3
import pickle
import os
from functools import lru_cache

class MemoryEntry:
    def __init__(self, dim: int):
        self.dim = dim
        self.weighted_sum = np.zeros(dim, dtype=np.int16) # int16 pour éviter l'overflow pendant l'apprentissage
        self.token_weights = {}
        self.bundle_cache = None # Packed version

    def update(self, token_hv: np.ndarray, weight: int = 1, token_name: str = None):
        # On travaille sur les bits déballés pour l'accumulation
        unpacked = np.unpackbits(token_hv)
        unpacked = unpacked.astype(np.int16)
        unpacked[unpacked == 0] = -1
        
        self.weighted_sum += unpacked * weight
        if token_name:
            self.token_weights[token_name] = self.token_weights.get(token_name, 0) + weight
        self.bundle_cache = None # Invalider le cache

    def get_bundle(self) -> np.ndarray:
        if self.bundle_cache is None:
            # Vote majoritaire : >0 devient 1, <=0 devient 0
            bits = (self.weighted_sum > 0).astype(np.uint8)
            self.bundle_cache = np.packbits(bits)
        return self.bundle_cache

class AssociativeMemory:
    def __init__(self, dim: int, db_path: str = "memory.nemdb", use_lsh: bool = True, max_ram_entries: int = 100000):
        self.dim = dim
        self.db_path = db_path
        self.use_lsh = use_lsh
        self.max_ram_entries = max_ram_entries
        
        # Initialisation DB
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA synchronous = NORMAL")
        # Mode Turbo D: SSD - On autorise SQLite a mapper 2Go et un gros cache
        self.conn.execute("PRAGMA mmap_size = 2147483648") 
        self.conn.execute("PRAGMA cache_size = -2000000") # 2Go de cache de pages
        self.conn.execute("CREATE TABLE IF NOT EXISTS storage (key BLOB PRIMARY KEY, data BLOB)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_key ON storage(key)")
        
        # Cache RAM pour la vitesse (LRU)
        self.ram_cache = {}
        self.write_buffer = {} # Clés modifiées à synchroniser

    def _hv_key(self, hv_packed: np.ndarray) -> bytes:
        return hv_packed.tobytes()

    def get_entry(self, hv_packed: np.ndarray) -> MemoryEntry:
        key = self._hv_key(hv_packed)
        
        # 1. Check Write Buffer (données les plus fraîches)
        if key in self.write_buffer:
            return self.write_buffer[key]

        # 2. Check RAM Cache
        if key in self.ram_cache:
            return self.ram_cache[key]
        
        # 3. Check Disk
        cursor = self.conn.execute("SELECT data FROM storage WHERE key = ?", (key,))
        row = cursor.fetchone()
        if row:
            entry = pickle.loads(row[0])
            self._update_ram_cache(key, entry)
            return entry
        
        # 4. Create New
        entry = MemoryEntry(self.dim)
        return entry

    def _update_ram_cache(self, key: bytes, entry: MemoryEntry):
        if len(self.ram_cache) >= self.max_ram_entries:
            self.ram_cache.clear()
        self.ram_cache[key] = entry

    def save_entry(self, hv_packed: np.ndarray, entry: MemoryEntry):
        key = self._hv_key(hv_packed)
        self.write_buffer[key] = entry
        
        # Si le buffer devient trop gros, on commit
        if len(self.write_buffer) >= 1000:
            self.commit()

    def commit(self):
        if not self.write_buffer:
            self.conn.commit()
            return
            
        # Écriture groupée
        items = [(k, pickle.dumps(v)) for k, v in self.write_buffer.items()]
        self.conn.executemany("INSERT OR REPLACE INTO storage (key, data) VALUES (?, ?)", items)
        self.conn.commit()
        
        # Transfert vers le cache RAM et vidage du buffer
        for k, v in self.write_buffer.items():
            self._update_ram_cache(k, v)
        self.write_buffer.clear()

    def learn_one_pass(self, q_hv_packed: np.ndarray, target_token: str, weight: int = 1):
        from hdc.representation import encode_token
        target_hv = encode_token(target_token, self.dim)
        
        entry = self.get_entry(q_hv_packed)
        entry.update(target_hv, weight, target_token)
        self.save_entry(q_hv_packed, entry)

    def predict_topk(self, query_hv_packed: np.ndarray, k: int = 5) -> list[str]:
        # On cherche d'abord le match exact sur disque/RAM
        entry = self.get_entry(query_hv_packed)
        
        if len(entry.token_weights) > 0:
            sorted_tokens = sorted(entry.token_weights.items(), key=lambda x: x[1], reverse=True)
            return [(t[0], 0) for t in sorted_tokens[:k]]
        
        # Si pas de match exact, le LSH (Optionnel ici, à ré-implémenter sur la DB si besoin)
        return []

    def close(self):
        self.conn.commit()
        self.conn.close()
