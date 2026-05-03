# Source Code - NemLM V5.3 Multi-Worker

Ce document contient les briques logicielles critiques du moteur NemLM dans sa version **Parallélisée (V5.3)**.

## 0. hdc/parallel_engine.py (Moteur Multi-Worker)
Gère l'entraînement asynchrone sur plusieurs cœurs avec filtrage des singletons.

```python
import multiprocessing as mp
from hdc.memory import AssociativeMemory
from hdc.representation import encode_context

def training_worker(task_queue, db_path, orders, dim):
    memory = AssociativeMemory(dim, db_path=db_path)
    seen_once = set() # Filtrage RAM
    
    while True:
        task = task_queue.get()
        if task is None: break
        
        context_tokens, target_token = task
        for n in orders:
            sub_context = context_tokens[-(n-1):] if n > 1 else []
            ngram_id = hash((tuple(sub_context), target_token))
            
            if ngram_id in seen_once:
                # Écriture uniquement à la 2ème occurrence
                hv = encode_context(sub_context, dim)
                memory.learn_one_pass(hv, target_token)
            else:
                seen_once.add(ngram_id)
    memory.commit()
```

## 1. hdc/representation.py (Primitives & Accumulateur)
Gère l'encodage binaire, les rotations et l'accumulation thématique.

```python
import numpy as np

DIM = 10000

def encode_token(token: str, dim: int = DIM) -> np.ndarray:
    seed = hash(token) % (2**32)
    rs = np.random.RandomState(seed)
    bits = rs.randint(0, 2, size=dim, dtype=np.uint8)
    return np.packbits(bits)

def rotate(hv_packed: np.ndarray, shift: int, dim: int = DIM) -> np.ndarray:
    bits = np.unpackbits(hv_packed)[:dim]
    rotated = np.roll(bits, shift)
    return np.packbits(rotated)

def encode_context(tokens: list[str], dim: int) -> np.ndarray:
    packed_result = np.zeros(dim // 8, dtype=np.uint8)
    for i, token in enumerate(reversed(tokens)):
        hv_packed = encode_token(token, dim)
        hv_pos_packed = rotate(hv_packed, i, dim)
        packed_result = np.bitwise_xor(packed_result, hv_pos_packed)
    return packed_result

def encode_context_n(tokens, n, dim):
    return encode_context(tokens[-n:], dim)

class ContextAccumulator:
    def __init__(self, dim: int = DIM, decay: float = 0.95):
        self.dim = dim
        self.decay = decay
        self.weighted_sum = np.zeros(dim, dtype=np.int16)
        
    def add(self, token_hv_packed: np.ndarray):
        bits = np.unpackbits(token_hv_packed)[:self.dim].astype(np.int16)
        bits[bits == 0] = -1
        self.weighted_sum = (self.weighted_sum * int(self.decay * 100)) // 100
        self.weighted_sum += bits
        
    def get_hv(self) -> np.ndarray:
        bits = (self.weighted_sum > 0).astype(np.uint8)
        return np.packbits(bits)

    def reset(self):
        self.weighted_sum.fill(0)
```

## 2. hdc/memory.py (Mémoire Associative & Backoff)
Gestion du stockage SQLite, du cache LRU et du vote pondéré.

```python
import numpy as np
import sqlite3
import pickle
import time
from collections import OrderedDict

class MemoryEntry:
    def __init__(self, dim: int):
        self.dim = dim
        self.weighted_sum = np.zeros(dim, dtype=np.int16)
        self.token_weights = {}

    def update(self, token_hv: np.ndarray, weight: int = 1, token_name: str = None):
        unpacked = np.unpackbits(token_hv).astype(np.int16)
        unpacked[unpacked == 0] = -1
        self.weighted_sum += unpacked * weight
        if token_name:
            self.token_weights[token_name] = self.token_weights.get(token_name, 0) + weight

class AssociativeMemory:
    def __init__(self, dim: int, db_path: str):
        self.dim = dim
        self.conn = sqlite3.connect(db_path)
        self.ram_cache = OrderedDict()
        self.write_buffer = {}
        self.last_commit_time = time.time()

    def predict_with_backoff(self, context_tokens: list[str], dim: int, k: int = 5) -> list[str]:
        total_scores = {}
        for n in [5, 4, 3, 2]:
            sub_context = context_tokens[-(n-1):] if n > 1 else []
            q_hv = encode_context(sub_context, dim)
            results = self.predict_topk(q_hv, k=k)
            if results:
                weight_factor = n ** 3
                for token, count in results:
                    total_scores[token] = total_scores.get(token, 0) + (count * weight_factor)
                if n == 5 and results[0][1] > 2: break
        return [t[0] for t in sorted(total_scores.items(), key=lambda x: x[1], reverse=True)[:k]]
```

## 3. hdc/attention.py (Darwinian Attention)
Fallback sémantique avec popcount et survie par utilité.

```python
import numpy as np

POPCOUNT_TABLE = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)

def hamming_batch(query_hv, keys_matrix):
    xor_res = np.bitwise_xor(keys_matrix, query_hv)
    return POPCOUNT_TABLE[xor_res].sum(axis=1)

class AttentionHead:
    def __init__(self, dim, n_keys):
        self.keys = np.zeros((n_keys, dim // 8), dtype=np.uint8)
        self.values = np.zeros((n_keys, dim // 8), dtype=np.uint8)
        self.hits = np.zeros(n_keys, dtype=np.uint32)

    def learn(self, context_hv, next_token_hv):
        if not self.full:
            idx = self.ptr; self.ptr += 1
        else:
            idx = np.argmin(self.hits)
        self.keys[idx] = context_hv
        self.values[idx] = next_token_hv
        self.hits[idx] = 1
```

## 4. hdc/v3_engine.py (Orchestrateur)
Fusion Local/Global et boucle AR.

```python
class V3Engine:
    def predict_next(self, context_tokens: list[str], top_k: int = 5) -> list[str]:
        # 1. Backoff (Local)
        exact_preds = self.memory.predict_with_backoff(context_tokens, self.dim, k=top_k)
        if exact_preds:
            self.accumulator.add(self.semantic.get_word_hv(exact_preds[0]))
            return exact_preds
            
        # 2. Attention (Thematique)
        l_hv = encode_context(context_tokens, self.dim)
        g_hv = self.accumulator.get_hv()
        query_hv = np.bitwise_xor(l_hv, g_hv)
        
        attention_hv = self.attention.forward(query_hv, k=8)
        return self.semantic.find_nearest_topk(attention_hv, k=top_k)
```
