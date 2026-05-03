import sqlite3
import pickle
import numpy as np
from hdc.v3_engine import V3Engine
from hdc.representation import DIM

class CompactInferenceEngine:
    def __init__(self, db_path='v2/memory_compact.nemdb'):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA mmap_size = 2147483648")
        self.conn.execute("PRAGMA cache_size = -2000000")
        self.dim = DIM

    def predict(self, context_hv_packed: np.ndarray):
        key = context_hv_packed.tobytes()
        cursor = self.conn.execute("SELECT data FROM storage WHERE key = ?", (key,))
        row = cursor.fetchone()
        if row:
            # On récupère (packed_sum, token_weights)
            _, weights = pickle.loads(row[0])
            sorted_tokens = sorted(weights.items(), key=lambda x: x[1], reverse=True)
            return [t[0] for t in sorted_tokens[:5]]
        return []

    def close(self):
        self.conn.close()

# Test rapide
engine = CompactInferenceEngine()
test_prompts = [
    "le projet de",
    "nous devons",
    "la commission européenne",
    "le président de la"
]

print("\n" + "="*40)
print("TEST D'INFÉRENCE NEMLM V3 (COMPACT)")
print("="*40)

from hdc.representation import encode_context

for prompt in test_prompts:
    tokens = prompt.lower().split()
    hv_packed = encode_context(tokens, DIM)
    preds = engine.predict(hv_packed)
    
    print(f"\nPrompt : '{prompt}'")
    print(f"Prédictions : {preds if preds else '??? (Contexte inconnu)'}")

engine.close()
