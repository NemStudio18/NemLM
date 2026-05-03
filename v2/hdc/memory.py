"""
HDC Associative Memory Layer V3 (Persistent & Bit-Packed)
Support pour le stockage sur disque via BTree (SQLite) pour gérer des millions de contextes sur 16GB.
"""
import numpy as np
import sqlite3
import pickle
import os
import time
from functools import lru_cache
from collections import OrderedDict

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
        
        # Cache RAM pour la vitesse (Vrai LRU)
        self.ram_cache = OrderedDict()
        self.write_buffer = {} # Clés modifiées à synchroniser
        self.last_commit_time = time.time()

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
        if key in self.ram_cache:
            self.ram_cache.move_to_end(key)
        self.ram_cache[key] = entry
        if len(self.ram_cache) > self.max_ram_entries:
            self.ram_cache.popitem(last=False) # Supprime le plus ancien (LRU)

    def save_entry(self, hv_packed: np.ndarray, entry: MemoryEntry):
        key = self._hv_key(hv_packed)
        self.write_buffer[key] = entry
        
        # Commit intelligent : toutes les 60s ou tous les 500 items
        now = time.time()
        if len(self.write_buffer) >= 500 or (now - self.last_commit_time) > 60:
            self.commit()
            self.last_commit_time = now

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

    def predict_topk(self, query_hv_packed: np.ndarray, k: int = 5) -> list[tuple[str, int]]:
        """Retourne les Top-K tokens et leurs poids."""
        entry = self.get_entry(query_hv_packed)
        if len(entry.token_weights) > 0:
            sorted_tokens = sorted(entry.token_weights.items(), key=lambda x: x[1], reverse=True)
            return sorted_tokens[:k]
        return []

    def predict_with_backoff(self, context_tokens: list[str], dim: int, k: int = 5) -> list[str]:
        """
        Implémente le backoff multi-échelle (HDC-Backoff).
        Interroge les ordres 5, 4, 3, 2 et fusionne les résultats avec pondération.
        """
        from hdc.representation import encode_context
        total_scores = {}
        
        # On teste les ordres du plus long au plus court
        for n in [5, 4, 3, 2]:
            sub_context = context_tokens[-(n-1):] if n > 1 else []
            if n > 1 and not sub_context: continue
            
            q_hv = encode_context(sub_context, dim)
            results = self.predict_topk(q_hv, k=k)
            
            if results:
                # Pondération : plus l'ordre est haut, plus le poids est fort (n^3)
                weight_factor = n ** 3
                for token, count in results:
                    total_scores[token] = total_scores.get(token, 0) + (count * weight_factor)
                
                # Optionnel : si on a un 5-gramme très fort, on peut s'arrêter là (Early Exit)
                if n == 5 and results[0][1] > 2: # Si on a vu ce 5-gramme plus de 2 fois
                    break

        if not total_scores:
            return []
            
        # Trier par score total accumulé
        sorted_final = sorted(total_scores.items(), key=lambda x: x[1], reverse=True)
        return [t[0] for t in sorted_final[:k]]

    def prune(self, min_count: int = 2):
        """Supprime les entrées trop rares pour réduire la taille de la DB et le bruit."""
        # Note: on doit itérer pour vérifier le total_count dans le pickle/JSON
        # Mais pour simplifier, on peut faire un scan
        print(f"[*] Elagage de la base (min_count={min_count})...")
        cursor = self.conn.execute("SELECT key, data FROM storage")
        to_delete = []
        for key, data in cursor:
            entry = pickle.loads(data)
            total = sum(entry.token_weights.values())
            if total < min_count:
                to_delete.append((key,))
        
        if to_delete:
            self.conn.executemany("DELETE FROM storage WHERE key = ?", to_delete)
            self.conn.commit()
            print(f"[OK] {len(to_delete)} entrees elaguees.")

    def close(self):
        self.commit()
        self.conn.close()
