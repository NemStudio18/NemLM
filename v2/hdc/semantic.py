"""
HDC Semantic Layer V3 (Bit-Packed)
"""
import numpy as np
from .representation import encode_token, rotate, DIM

class SemanticIndex:
    def __init__(self, dim: int = DIM):
        self.dim = dim
        self.vocabulary = {} # token -> hv_packed
        self.id_to_token = {}
        self.token_hvs = [] # List of packed HVs for search

    def get_word_hv(self, token: str) -> np.ndarray:
        if token not in self.vocabulary:
            hv = encode_token(token, self.dim)
            idx = len(self.token_hvs)
            self.vocabulary[token] = hv
            self.id_to_token[idx] = token
            self.token_hvs.append(hv)
        return self.vocabulary[token]

    def find_nearest_topk(self, query_hv_packed: np.ndarray, k: int = 5) -> list[str]:
        if not self.token_hvs: return ["???"]
        
        hvs_matrix = np.stack(self.token_hvs)
        xor_res = np.bitwise_xor(hvs_matrix, query_hv_packed)
        diff_bits = np.unpackbits(xor_res, axis=1).sum(axis=1)
        
        top_k_idx = np.argsort(diff_bits)[:k]
        return [self.id_to_token[idx] for idx in top_k_idx]

    def find_nearest(self, query_hv_packed: np.ndarray) -> str:
        if not self.token_hvs: return "???"
        
        # On empile les HVs pour la recherche vectorisee
        hvs_matrix = np.stack(self.token_hvs)
        
        # Hamming distance vectorisee sur bits packes (XOR + count_nonzero)
        xor_res = np.bitwise_xor(hvs_matrix, query_hv_packed)
        # On unpack pour compter les bits a 1
        diff_bits = np.unpackbits(xor_res, axis=1).sum(axis=1)
        
        best_idx = np.argmin(diff_bits)
        return self.id_to_token[best_idx]

    def encode_bow(self, tokens: list[str]) -> np.ndarray:
        if not tokens: return np.zeros(self.dim // 8, dtype=np.uint8)
        hvs_bits = [np.unpackbits(self.get_word_hv(t)) for t in tokens]
        sum_hvs = np.sum(hvs_bits, axis=0)
        bits = (sum_hvs > len(tokens) // 2).astype(np.uint8)
        return np.packbits(bits)
