"""
HDC Memory Layer V2 (Optimisée)
- Majority Bundling incremental.
- Passive-Aggressive Binary Perceptron.
- Weighted Sum vector maintenu en temps réel pour O(D) bundling.
"""

import numpy as np
from .representation import encode_context, encode_token, hamming
from .lsh import LSHIndex

class MemoryEntry:
    def __init__(self, dim: int):
        # Somme pondérée des vecteurs (1 -> +w, 0 -> -w)
        # Permet de calculer le bundle majoritaire en O(D)
        self.weighted_sum = np.zeros(dim, dtype=np.int32)
        self.bundle_cache = None
        self.token_weights = {} # token -> weight

    def update(self, token: str, dim: int, delta: int):
        """Mise à jour incrémentale de la somme pondérée."""
        if delta == 0: return
        
        token_hv = encode_token(token, dim)
        # (token_hv * 2 - 1) transforme [0, 1] en [-1, 1]
        influence = (token_hv.astype(np.int32) * 2 - 1) * delta
        self.weighted_sum += influence
        
        # Mise à jour du poids
        self.token_weights[token] = self.token_weights.get(token, 0) + delta
        # Invalide le cache
        self.bundle_cache = None

class AssociativeMemory:
    def __init__(self, dim: int, use_lsh: bool = True):
        self.dim = dim
        self.use_lsh = use_lsh
        self.storage: dict[int, MemoryEntry] = {}
        self._lsh = None
        self._vocab_list = []
        self._vocab_matrix = None

    def _hv_key(self, hv: np.ndarray) -> int:
        packed = np.packbits(hv)
        return hash(packed.tobytes()) & 0xFFFFFFFFFFFFFFFF

    def learn_one_pass(self, context_tokens: list[str], next_token: str):
        """Phase 1 : Accumulation (Bundling initial)."""
        ctx_hv = encode_context(context_tokens, self.dim)
        key = self._hv_key(ctx_hv)
        
        if key not in self.storage:
            self.storage[key] = MemoryEntry(self.dim)
        
        # Bundling initial : delta = 1
        self.storage[key].update(next_token, self.dim, 1)

    def get_bundle(self, key: int) -> np.ndarray:
        entry = self.storage[key]
        if entry.bundle_cache is not None:
            return entry.bundle_cache
        
        # Vote majoritaire : 1 si somme > 0, 0 sinon
        bundle = (entry.weighted_sum > 0).astype(np.uint8)
        entry.bundle_cache = bundle
        return bundle

    def update_passive_aggressive(self, context_tokens: list[str], true_token: str, predicted_token: str, margin: int = 500):
        """
        Ajustement Passive-Aggressive :
        Si distance(true) > distance(pred) - margin:
           Renforce true (+delta), pénalise pred (-delta)
        """
        ctx_hv = encode_context(context_tokens, self.dim)
        key = self._hv_key(ctx_hv)
        entry = self.storage.get(key)
        if not entry: return

        # On calcule les distances réelles pour ce contexte
        bundle = self.get_bundle(key)
        dist_true = hamming(bundle, encode_token(true_token, self.dim))
        dist_pred = hamming(bundle, encode_token(predicted_token, self.dim)) if predicted_token else self.dim
        
        # Condition PA : si la confiance n'est pas assez grande
        if dist_true > dist_pred - margin:
            # Facteur d'agressivité (delta)
            # Plus l'erreur est grande, plus on ajuste (ici simple delta fixe pour rester binaire/entier)
            delta = 2
            entry.update(true_token, self.dim, delta)
            if predicted_token:
                entry.update(predicted_token, self.dim, -delta // 2)

    def build_lsh(self, vocabulary: list[str], num_tables: int = 10, num_bits: int = 16):
        self._vocab_list = vocabulary
        self._lsh = LSHIndex(self.dim, num_tables=num_tables, num_bits=num_bits)
        self._lsh.build(vocabulary, lambda t: encode_token(t, self.dim))

    def predict_topk(self, context_tokens: list[str], vocabulary: list[str], k: int = 5) -> list[tuple[str, int]]:
        ctx_hv = encode_context(context_tokens, self.dim)
        key = self._hv_key(ctx_hv)
        
        bundle = self.get_bundle(key) if key in self.storage else ctx_hv
            
        if self.use_lsh and self._lsh:
            return self._lsh.query(bundle, k=k)
        
        # Scan exact
        if self._vocab_matrix is None or len(self._vocab_list) != len(vocabulary):
            self._vocab_matrix = np.stack([encode_token(t, self.dim) for t in vocabulary])
            self._vocab_list = vocabulary
            
        diffs = np.bitwise_xor(self._vocab_matrix, bundle[np.newaxis, :])
        distances = np.count_nonzero(diffs, axis=1)
        top_k_idx = np.argsort(distances)[:k]
        return [(vocabulary[i], int(distances[i])) for i in top_k_idx]

    @property
    def size(self) -> int:
        return len(self.storage)

    @property
    def memory_bytes(self) -> int:
        return self.size * (self.dim * 4 + 512) # int32 = 4 bytes
