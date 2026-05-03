"""
NemLM Compact Inference Engine (V5.3)
Moteur optimisé pour la lecture seule (Read-Only) et la vitesse extrême.
"""
import sqlite3
import pickle
import numpy as np
from hdc.representation import encode_context

class CompactMemory:
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        # Optimisations lecture seule
        self.conn.execute("PRAGMA query_only = ON")
        self.conn.execute("PRAGMA mmap_size = 536870912") # 512 Mo mmap
        self.conn.execute("PRAGMA cache_size = -100000") # 100 Mo cache
        
    def get_preds(self, hv_packed: np.ndarray) -> list[str]:
        key = hv_packed.tobytes()
        cursor = self.conn.execute("SELECT preds FROM distilled WHERE id = ?", (key,))
        row = cursor.fetchone()
        if row:
            return pickle.loads(row[0])
        return []

class CompactEngine:
    def __init__(self, db_path: str, dim: int = 10000):
        self.dim = dim
        self.memory = CompactMemory(db_path)
        
    def predict_next(self, context_tokens: list[str], top_k: int = 5) -> list[str]:
        # Backoff Multi-échelle (Version Compacte)
        total_scores = {}
        
        for n in [5, 4, 3, 2]:
            sub_context = context_tokens[-(n-1):] if n > 1 else []
            q_hv = encode_context(sub_context, self.dim)
            
            preds = self.memory.get_preds(q_hv)
            if preds:
                weight_factor = n ** 3
                for i, token in enumerate(preds):
                    # On donne plus de poids au premier de la liste
                    rank_weight = (top_k - i) 
                    total_scores[token] = total_scores.get(token, 0) + (rank_weight * weight_factor)
                
                # Early exit si on a un match fort sur un n-gramme long
                if n >= 4:
                    break
                    
        if not total_scores:
            return ["<unk>"]
            
        return [t[0] for t in sorted(total_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]]
