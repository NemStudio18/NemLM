"""
HDC Semantic Layer V2 - Multi-Layer Ranking
Implémente une recherche à deux niveaux :
1. Recherche grossière (BoW) -> Sélection Top-N
2. Re-ranking (Séquentiel) -> Sélection finale
"""

import numpy as np
import re
from .representation import encode_token, rotate, DIM
from .lsh import LSHIndex

class SemanticIndex:
    def __init__(self, dim: int = DIM, use_lsh: bool = True, lsh_bits: int = 12):
        self.dim = dim
        self.use_lsh = use_lsh
        self.lsh = LSHIndex(dim, num_tables=10, num_bits=lsh_bits) if use_lsh else None
        self.items = [] 
        self.hvs_bow = None  # Layer 1: Bag of Words
        self.hvs_seq = None  # Layer 2: Séquentiel (avec rotations)

    def clean_text(self, text: str) -> str:
        return re.sub(r"[^a-z0-9\s]", " ", text.lower()).strip()

    def encode_bow(self, tokens: list[str]) -> np.ndarray:
        """Encode une phrase sans ordre (BoW). Rapide pour le filtrage."""
        if not tokens:
            return np.zeros(self.dim, dtype=np.uint8)
        hvs = [encode_token(t, self.dim) for t in tokens]
        sum_hvs = np.sum(hvs, axis=0, dtype=np.int32)
        return (sum_hvs > len(hvs) // 2).astype(np.uint8)

    def encode_sequence(self, tokens: list[str]) -> np.ndarray:
        """Encode une phrase avec ordre (Rotations). Précis pour le re-ranking."""
        if not tokens:
            return np.zeros(self.dim, dtype=np.uint8)
        result = np.zeros(self.dim, dtype=np.uint8)
        for i, token in enumerate(tokens):
            hv = encode_token(token, self.dim)
            # Rotation basée sur la position dans la phrase
            hv_pos = rotate(hv, i)
            result = np.bitwise_xor(result, hv_pos)
        return result

    def build(self, sentences: list[str]):
        self.items = sentences
        print(f"  - Génération des HVs BoW et Séquentiels pour {len(sentences)} phrases...")
        
        bow_list = []
        seq_list = []
        for s in sentences:
            tokens = self.clean_text(s).split()
            bow_list.append(self.encode_bow(tokens))
            seq_list.append(self.encode_sequence(tokens))
            
        self.hvs_bow = np.stack(bow_list)
        self.hvs_seq = np.stack(seq_list)
        
        if self.use_lsh:
            # On indexe le BoW dans le LSH car il est plus stable pour la recherche initiale
            self.lsh.build_precomputed(sentences, self.hvs_bow)

    def query(self, question: str, k: int = 3, rerank_top_n: int = 20) -> list[tuple[str, int, float]]:
        """
        Recherche Multi-Couche :
        1. Retrieval (LSH ou Scan BoW) -> Top N
        2. Re-ranking (Hamming sur Seq) -> Top K
        """
        tokens = self.clean_text(question).split()
        q_bow = self.encode_bow(tokens)
        q_seq = self.encode_sequence(tokens)
        
        # --- Étape 1 : Coarse Search (BoW) ---
        candidates_idx = []
        if self.use_lsh:
            # Récupération des indices des candidats via LSH
            candidates = self.lsh.query_indices(q_bow, k=rerank_top_n)
            candidates_idx = candidates if candidates else range(len(self.items))
        else:
            # Scan exact BoW pour le filtrage
            diffs = np.bitwise_xor(self.hvs_bow, q_bow[np.newaxis, :])
            dist_bow = np.count_nonzero(diffs, axis=1)
            candidates_idx = np.argsort(dist_bow)[:rerank_top_n]

        # --- Étape 2 : Re-ranking (Séquentiel) ---
        # On ne compare que les candidats sélectionnés
        final_scores = []
        for idx in candidates_idx:
            # Distance Hamming sur le HV séquentiel
            dist_seq = np.count_nonzero(np.bitwise_xor(self.hvs_seq[idx], q_seq))
            # Distance Hamming sur le BoW (pour combiner les scores)
            dist_bow = np.count_nonzero(np.bitwise_xor(self.hvs_bow[idx], q_bow))
            
            # Score combiné (pondéré) : 30% BoW, 70% Séquence
            combined_dist = (0.3 * dist_bow + 0.7 * dist_seq)
            final_scores.append((idx, combined_dist))
            
        # Tri final
        final_scores.sort(key=lambda x: x[1])
        top_k = final_scores[:k]
        
        return [(self.items[idx], int(score), (1 - score/self.dim)*100) for idx, score in top_k]
