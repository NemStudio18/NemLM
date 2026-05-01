"""
HDC Memory Layer V2 (Optimisée)
- Majority Bundling incremental.
- Passive-Aggressive Binary Perceptron.
- Weighted Sum vector maintenu en temps réel pour O(D) bundling.
"""

import numpy as np
from .representation import encode_context, encode_token, hamming
from .lsh import LSHIndex
from .persistence import save_memory, load_memory

class MemoryEntry:
    def __init__(self, dim: int):
        self.weighted_sum = np.zeros(dim, dtype=np.int32)
        self.bundle_cache = None
        self.token_weights = {} # token -> weight

    def update(self, token: str, dim: int, delta: int):
        if delta == 0: return
        token_hv = encode_token(token, dim)
        influence = (token_hv.astype(np.int32) * 2 - 1) * delta
        self.weighted_sum += influence
        self.token_weights[token] = self.token_weights.get(token, 0) + delta
        self.bundle_cache = None

    def merge(self, other: 'MemoryEntry'):
        """Fusionne les poids d'une autre entrée (Somme vectorielle)."""
        self.weighted_sum += other.weighted_sum
        for token, weight in other.token_weights.items():
            self.token_weights[token] = self.token_weights.get(token, 0) + weight
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

    def learn_one_pass(self, context, next_token: str):
        """Phase 1 : Accumulation. context peut être une liste de tokens ou un HV."""
        if isinstance(context, list):
            ctx_hv = encode_context(context, self.dim)
        else:
            ctx_hv = context
            
        key = self._hv_key(ctx_hv)
        if key not in self.storage:
            self.storage[key] = MemoryEntry(self.dim)
        self.storage[key].update(next_token, self.dim, 1)

    def get_bundle(self, key: int) -> np.ndarray:
        entry = self.storage[key]
        if entry.bundle_cache is not None:
            return entry.bundle_cache
        bundle = (entry.weighted_sum > 0).astype(np.uint8)
        entry.bundle_cache = bundle
        return bundle

    def update_passive_aggressive(self, context, true_token: str, predicted_token: str = None, margin: int = 500):
        """Ajustement PA. context peut être une liste de tokens ou un HV."""
        if isinstance(context, list):
            ctx_hv = encode_context(context, self.dim)
        else:
            ctx_hv = context
            
        key = self._hv_key(ctx_hv)
        entry = self.storage.get(key)
        if not entry: 
            # Si le contexte n'existe pas encore, on l'initialise
            self.learn_one_pass(ctx_hv, true_token)
            return

        bundle = self.get_bundle(key)
        dist_true = hamming(bundle, encode_token(true_token, self.dim))
        dist_pred = hamming(bundle, encode_token(predicted_token, self.dim)) if predicted_token else self.dim
        
        if dist_true > dist_pred - margin:
            delta = 2
            entry.update(true_token, self.dim, delta)
            if predicted_token:
                entry.update(predicted_token, self.dim, -delta // 2)

    def build_lsh(self, vocabulary: list[str], num_tables: int = 10, num_bits: int = 16):
        self._vocab_list = vocabulary
        self._lsh = LSHIndex(self.dim, num_tables=num_tables, num_bits=num_bits)
        self._lsh.build(vocabulary, lambda t: encode_token(t, self.dim))

    def predict_topk(self, context, vocabulary: list[str] = None, k: int = 5) -> list[str]:
        """Retourne les k tokens les plus probables."""
        if isinstance(context, list):
            ctx_hv = encode_context(context, self.dim)
        else:
            ctx_hv = context
            
        key = self._hv_key(ctx_hv)
        bundle = self.get_bundle(key) if key in self.storage else ctx_hv
            
        if self.use_lsh and self._lsh:
            res = self._lsh.query(bundle, k=k)
            return [r[0] for r in res]
        
        if self._vocab_matrix is None or len(self._vocab_list) != len(vocabulary):
            self._vocab_list = vocabulary
            # On stocke la matrice en format signe (-1, 1) pour np.dot
            self._vocab_matrix = (np.stack([encode_token(t, self.dim) for t in vocabulary]).astype(np.int8) * 2 - 1)
            
        # Distance de Hamming via produit matriciel (tres rapide)
        # Dist = (D - dot(A_signed, B_signed)) / 2
        q_signed = (bundle.astype(np.int8) * 2 - 1)
        scores = np.dot(self._vocab_matrix, q_signed)
        distances = (self.dim - scores) // 2
        
        top_k_idx = np.argsort(distances)[:k]
        return [vocabulary[i] for i in top_k_idx]

    def merge(self, other: 'AssociativeMemory'):
        """Fusionne une autre mémoire associative dans celle-ci."""
        for key, other_entry in other.storage.items():
            if key not in self.storage:
                self.storage[key] = MemoryEntry(self.dim)
            self.storage[key].merge(other_entry)

    def save(self, path: str):
        save_memory(path, self.storage, self.dim)

    def load(self, path: str):
        loaded_storage, dim = load_memory(path)
        if loaded_storage:
            self.storage = loaded_storage
            self.dim = dim
            return True
        return False

    @property
    def size(self) -> int:
        return len(self.storage)
