# Source Code - NemLM V5.3 (HDC-AR High-Fidelity)

Ce document contient les briques logicielles critiques du moteur NemLM dans sa version **Industrialisée (V5.3)**.

## 1. hdc/compact_engine.py (Inférence Haute Fidélité)
Optimisé pour la vitesse et le fallback sémantique total.

```python
class CompactEngine:
    def predict_next(self, context_tokens: list[str], top_k: int = 5) -> list[str]:
        # 1. Backoff Multi-échelle (Précision Syntaxique)
        total_scores = {}
        for n in [5, 4, 3, 2]:
            sub_context = context_tokens[-(n-1):] if n > 1 else []
            q_hv = encode_context(sub_context, self.dim)
            preds = self.memory.get_preds(q_hv)
            if preds:
                # Pondération V3 (n^4) pour écraser le bruit
                weight_factor = n ** 4
                for token, count in preds:
                    total_scores[token] = total_scores.get(token, 0) + (count * weight_factor)
                
                # Early Exit (Count > 2 sur 5-gramme)
                if n == 5 and preds[0][1] > 2: break

        # 2. Fallback Attention Sémantique (Cohérence Globale)
        l_hv = encode_context(context_tokens, self.dim)
        g_hv = self.accumulator.get_hv()
        query_hv = np.bitwise_xor(l_hv, g_hv)
        
        attn_hv = self.memory.attention.forward(query_hv, k=8)
        return self.semantic.find_nearest_topk(attn_hv, k=top_k)
```

## 2. hdc/representation.py (Primitives & Accumulateur)
Gère l'encodage binaire, les rotations et l'accumulation thématique.

```python
def encode_context(tokens: list[str], dim: int) -> np.ndarray:
    packed_result = np.zeros(dim // 8, dtype=np.uint8)
    for i, token in enumerate(reversed(tokens)):
        hv_packed = encode_token(token, dim)
        hv_pos_packed = rotate(hv_packed, i, dim)
        packed_result = np.bitwise_xor(packed_result, hv_pos_packed)
    return packed_result

class ContextAccumulator:
    def add(self, token_hv_packed: np.ndarray):
        bits = np.unpackbits(token_hv_packed)[:self.dim].astype(np.int16)
        bits[bits == 0] = -1
        # Decay sémantique de 5% par token
        self.weighted_sum = (self.weighted_sum * 95) // 100
        self.weighted_sum += bits
```

## 3. hdc/attention.py (Darwinian Attention)
Mécanisme de repli utilisant la distance de Hamming et la survie par utilité.

```python
def hamming_batch(query_hv, keys_matrix):
    """Calcul ultra-rapide via table de lookup."""
    xor_res = np.bitwise_xor(keys_matrix, query_hv)
    return POPCOUNT_TABLE[xor_res].sum(axis=1)

class AttentionHead:
    def learn(self, context_hv, next_token_hv):
        if self.full:
            # Darwinisme : on écrase le souvenir le moins utile (min hits)
            idx = np.argmin(self.hits)
        else:
            idx = self.ptr; self.ptr += 1
        self.keys[idx] = context_hv
        self.values[idx] = next_token_hv
        self.hits[idx] = 1
```

## 4. hdc/parallel_engine.py (Entraînement Turbo)
Entraînement asynchrone multi-processus avec workers spécialisés.

```python
def training_worker(task_queue, db_path, orders, dim):
    memory = AssociativeMemory(dim, db_path=db_path)
    while True:
        task = task_queue.get()
        if task is None: break
        context_tokens, target_token = task
        # Apprentissage multi-ordre sans filtrage pour la fidélité
        for n in orders:
            hv = encode_context(context_tokens[-(n-1):], dim)
            memory.learn_one_pass(hv, target_token)
```

## 5. hdc/reasoning.py (Reasoning Accumulator - Phase 4)
Mécanisme émergent de réflexion par convergence Hamming.

```python
def reason(self, question: str, max_steps: int = 5):
    q_hv = encode_context(question.split(), self.dim)
    self.accumulator.reset()
    self.accumulator.add(q_hv)
    
    for step in range(max_steps):
        current_hv = self.accumulator.get_hv()
        # Retrieval multi-step
        retrieval = self.predict_next_from_hv(current_hv)
        if not retrieval: break
        
        # Injection du résultat pour orienter le prochain step
        self.accumulator.add(encode_token(retrieval[0], self.dim))
        
        # Convergence : si le nouveau contexte est proche du précédent
        if hamming(current_hv, self.accumulator.get_hv()) < self.dim // 8:
            break
```

## 6. hdc/binary_layers.py (Binary Transformer - Phase 3B)
Cœur de l'apprentissage différentiable binaire (STE Backprop).

```python
class BinaryLinear:
    def forward(self, x):
        # Binarisation des poids latents float32 -> binaire (-1, 1)
        w_mean = np.mean(self.weights_latent)
        self.w_bin = np.sign(self.weights_latent - w_mean)
        return np.dot(x, self.w_bin.T) + self.bias

    def backward(self, x, grad_output, lr=0.01):
        # Straight-Through Estimator (STE) + Gradient Clipping
        grad_w = np.outer(grad_output, x)
        norm = np.linalg.norm(grad_w)
        if norm > 1.0: grad_w /= norm # Stabilisation
        self.weights_latent -= lr * grad_w

## 7. train_bt_v1.py (Prototype Validation)
Script de validation du cœur différentiable :
- **Mode Ultra-Light** : 512-dim pour convergence rapide.
- **Déterminisme** : Utilisation de HVs déterministes par hash pour le prototype.
- **Logging** : Suivi persistent dans `result_tests/bt_prototype_log.txt`.

---
*NemStudio - Advanced Agentic Coding Project*
