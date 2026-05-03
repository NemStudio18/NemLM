"""
NemLM Compact Inference Engine (V5.3 HDC-AR)
Moteur optimis\u00e9 pour la lecture seule (Read-Only) et la fid\u00e9lit\u00e9 scientifique.
"""
import sqlite3
import pickle
import numpy as np
from hdc.representation import encode_context, DIM, ContextAccumulator
from hdc.attention import MultiHeadBinaryAttention
from hdc.semantic import SemanticIndex

class CompactMemory:
    def __init__(self, db_path: str, dim: int):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        # Optimisations lecture seule
        self.conn.execute("PRAGMA query_only = ON")
        self.conn.execute("PRAGMA mmap_size = 536870912") # 512 Mo mmap
        
        # Chargement de l'Attention
        self.attention = MultiHeadBinaryAttention(dim=dim)
        self._load_attention()
        
    def _load_attention(self):
        """Charge les t\u00eates d'attention depuis la table d\u00e9di\u00e9e."""
        total_keys = 0
        cursor = self.conn.execute("SELECT head_id, data FROM attention")
        rows = cursor.fetchall()
        for head_id, data in rows:
            head_data = pickle.loads(data)
            self.attention.heads[head_id].from_dict(head_data)
            total_keys += head_data["ptr"] if not head_data["full"] else self.attention.n_keys
        if total_keys > 0:
            print(f"[*] Attention compacte charg\u00e9e : {total_keys} souvenirs.")

    def get_preds(self, hv_packed: np.ndarray) -> list[tuple[str, int]]:
        key = hv_packed.tobytes()
        # Nouvelle requ\u00eate pour le sch\u00e9ma multi-lignes
        cursor = self.conn.execute("SELECT token, weight FROM distilled WHERE context_hv = ? ORDER BY weight DESC", (key,))
        return cursor.fetchall()

class CompactEngine:
    def __init__(self, db_path: str, dim: int = DIM):
        self.dim = dim
        self.memory = CompactMemory(db_path, dim)
        self.semantic = SemanticIndex(dim)
        self.accumulator = ContextAccumulator(dim=dim, decay=0.95)
        
    def reset_context(self):
        self.accumulator.reset()
        
    def predict_next(self, context_tokens: list[str], top_k: int = 5) -> list[str]:
        # 1. Backoff Multi-échelle (Local)
        total_scores = {}
        
        for n in [5, 4, 3, 2]:
            sub_context = context_tokens[-(n-1):] if n > 1 else []
            q_hv = encode_context(sub_context, self.dim)
            
            preds = self.memory.get_preds(q_hv)
            if preds:
                # Pondération spécification V3 (n**4)
                weight_factor = n ** 4
                for token, count in preds:
                    total_scores[token] = total_scores.get(token, 0) + (count * weight_factor)
                
                # Early Exit sur 5-gramme robuste : Protection contre le bruit statistique
                if n == 5 and preds[0][1] > 2:
                    break
        
        # SI on a trouvé des résultats dans le backoff -> RETOUR IMMEDIAT
        if total_scores:
            sorted_preds = [t[0] for t in sorted(total_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]]
            # Mise à jour accumulator
            self.accumulator.add(self.semantic.get_word_hv(sorted_preds[0]))
            return sorted_preds
            
        # 2. Fallback Attention Sémantique (Uniquement si backoff muet)
        l_hv = encode_context(context_tokens[-5:], self.dim)
        g_hv = self.accumulator.get_hv()
        query_hv = np.bitwise_xor(l_hv, g_hv)
        
        attn_hv = self.memory.attention.forward(query_hv, k=8)
        predicted_tokens = self.semantic.find_nearest_topk(attn_hv, k=top_k)
        
        if predicted_tokens:
            self.accumulator.add(self.semantic.get_word_hv(predicted_tokens[0]))
            return predicted_tokens
            
        return ["<unk>"]
