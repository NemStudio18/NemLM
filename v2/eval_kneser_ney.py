"""
Benchmark Comparatif : NemLM (HDC) vs Kneser-Ney (N-Grammes)
Tache : Next Word Prediction (Cloze Test) sur Europarl FR.
"""
import time
import math
from collections import Counter, defaultdict

class KneserNeyModel:
    """Modele N-gramme classique avec lissage Kneser-Ney."""
    def __init__(self, n=5, discount=0.75):
        self.n = n
        self.discount = discount
        self.counts = [Counter() for _ in range(n + 1)]
        self.contexts = [Counter() for _ in range(n + 1)]

    def train(self, sentences):
        print(f"Entrainement Kneser-Ney {self.n}-grammes...")
        for sent in sentences:
            for i in range(len(sent)):
                for order in range(1, self.n + 1):
                    if i + order <= len(sent):
                        ngram = tuple(sent[i:i+order])
                        self.counts[order][ngram] += 1
                        context = ngram[:-1]
                        self.contexts[order][context] += 1

    def predict_topk(self, context_words, k=5):
        # Implementation simplifiee du backoff pour la comparaison
        context = tuple(context_words[-(self.n-1):])
        
        # On cherche dans les n-grammes de taille decroissante
        for order in range(len(context) + 1, 0, -1):
            curr_ctx = context[-(order-1):] if order > 1 else ()
            candidates = []
            ctx_count = self.contexts[order][curr_ctx]
            
            if ctx_count > 0:
                # Trouve tous les mots qui suivent ce contexte
                for ngram, count in self.counts[order].items():
                    if ngram[:-1] == curr_ctx:
                        candidates.append((ngram[-1], count / ctx_count))
                
                if candidates:
                    candidates.sort(key=lambda x: x[1], reverse=True)
                    return [c[0] for c in candidates[:k]]
        return []

# --- Script de benchmark sera ajoute ici ---
