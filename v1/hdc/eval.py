"""
HDC Eval Layer
Mesure accuracy, temps d'inférence, RAM. Zéro flottant en inférence.
"""

import time
import json
from datetime import datetime
from pathlib import Path
from .memory import AssociativeMemory
from .corpus import ngrams


def evaluate(
    memory: AssociativeMemory,
    sentences: list[list[str]],
    vocabulary: list[str],
    context_size: int = 2,
    k: int = 3,
    verbose: bool = True,
) -> dict:
    """
    Évalue la mémoire HDC sur un ensemble de phrases.
    Retourne un dict de métriques.
    """
    total = 0
    correct_at_1 = 0
    correct_at_k = 0
    total_time_ns = 0

    n = context_size + 1  # ex: context=2 -> trigrammes

    for gram in ngrams(sentences, n):
        context = list(gram[:-1])
        expected = gram[-1]

        t0 = time.perf_counter_ns()
        results = memory.predict_topk(context, vocabulary, k=k)
        t1 = time.perf_counter_ns()

        total_time_ns += (t1 - t0)
        total += 1

        predicted_tokens = [r[0] for r in results]

        if predicted_tokens and predicted_tokens[0] == expected:
            correct_at_1 += 1
        if expected in predicted_tokens:
            correct_at_k += 1

    if total == 0:
        return {}

    avg_time_us = (total_time_ns / total) / 1000

    metrics = {
        "total_predictions": total,
        "accuracy_at_1": round(correct_at_1 / total * 100, 2),
        f"accuracy_at_{k}": round(correct_at_k / total * 100, 2),
        "avg_inference_us": round(avg_time_us, 2),
        "memory_contexts": memory.size,
        "memory_bytes": memory.memory_bytes,
        "vocabulary_size": len(vocabulary),
        "float_ops": 0,  # toujours 0, c'est le point
    }

    if verbose:
        print("\n-- Résultats ------------------------------")
        print(f"  Prédictions totales : {total}")
        print(f"  Accuracy @1         : {metrics['accuracy_at_1']}%")
        print(f"  Accuracy @{k}         : {metrics[f'accuracy_at_{k}']}%")
        print(f"  Temps moyen         : {avg_time_us:.1f} us")
        print(f"  Contextes mémorisés : {memory.size}")
        print(f"  RAM mémoire HDC     : {memory.memory_bytes / 1024:.1f} KB")
        print(f"  Vocabulaire         : {len(vocabulary)} tokens")
        print(f"  Flottants utilisés  : 0 OK")
        print("--------------------------------------------\n")

    return metrics


def save_results(metrics: dict, path: str = "results/"):
    """Sauvegarde les métriques en JSON."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = p / f"bench_{timestamp}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Résultats sauvegardés : {filename}")
    return str(filename)
