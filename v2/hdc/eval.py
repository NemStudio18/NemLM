"""
HDC Eval Layer V2
- Accuracy@1, @3, @5.
- Perplexité (estimée via distance Hamming inverse).
- Temps d'inférence.
"""

import time
import numpy as np
from .memory import AssociativeMemory
from .corpus import ngrams

def compute_perplexity(distances: list[int], dim: int) -> float:
    """
    Estime la perplexité.
    En HDC, on n'a pas de probabilités log-softmax.
    On approxime P(token) ~ exp(- Hamming / sigma).
    """
    if not distances: return 0.0
    # On normalise les distances (0 à 1)
    norm_dist = np.array(distances) / dim
    # Plus la distance est petite, plus la proba est grande
    # Une distance de 0.5 (aléatoire) donne une proba faible
    probs = np.exp(-10 * norm_dist) # Heuristique pour transformer distance en "confiance"
    probs /= np.sum(probs) if np.sum(probs) > 0 else 1.0
    
    # Perplexité = exp(- moyenne(log(P_cible)))
    # Ici on prend juste la proba du top 1 ou une moyenne
    log_probs = np.log(probs + 1e-10)
    return float(np.exp(-np.mean(log_probs)))

def evaluate(
    memory: AssociativeMemory,
    sentences: list[list[str]],
    vocabulary: list[str],
    context_size: int = 2,
    verbose: bool = True
) -> dict:
    total = 0
    correct = {1: 0, 3: 0, 5: 0}
    latencies = []
    all_top_distances = []

    for gram in ngrams(sentences, context_size + 1):
        context = list(gram[:-1])
        expected = gram[-1]
        
        t0 = time.perf_counter()
        results = memory.predict_topk(context, vocabulary, k=5)
        t1 = time.perf_counter()
        
        latencies.append(t1 - t0)
        total += 1
        
        tokens = [r[0] for r in results]
        if results:
            all_top_distances.append(results[0][1])
            
        for k in [1, 3, 5]:
            if expected in tokens[:k]:
                correct[k] += 1

    if total == 0: return {}

    metrics = {
        "accuracy_at_1": round(correct[1] / total * 100, 2),
        "accuracy_at_3": round(correct[3] / total * 100, 2),
        "accuracy_at_5": round(correct[5] / total * 100, 2),
        "avg_latency_us": round(np.mean(latencies) * 1e6, 2),
        "perplexity": round(compute_perplexity(all_top_distances, memory.dim), 2),
        "total": total
    }

    if verbose:
        print(f"\n-- Eval Results (N={total}) --")
        print(f"  Acc@1: {metrics['accuracy_at_1']}%")
        print(f"  Acc@3: {metrics['accuracy_at_3']}%")
        print(f"  Acc@5: {metrics['accuracy_at_5']}%")
        print(f"  PPL:   {metrics['perplexity']}")
        print(f"  Time:  {metrics['avg_latency_us']} us")
        print("----------------------------")

    return metrics
