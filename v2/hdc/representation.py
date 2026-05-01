"""
HDC Representation Layer V2
- Dimension jusqu'à 100 000 bits.
- Rotation positionnelle par nombres premiers.
- Encodage par n-grammes de caractères (subwords).
- Cache LRU pour les tokens.
"""

import hashlib
import numpy as np
from functools import lru_cache

# Configuration par défaut
DIM = 30000

# Nombres premiers pour les rotations positionnelles
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]

# Cache pour les vecteurs de tokens
_TOKEN_CACHE = {}

def _seed_from_str(s: str) -> int:
    h = hashlib.sha256(s.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big")

def generate_random_hv(seed_str: str, dim: int) -> np.ndarray:
    rng = np.random.default_rng(_seed_from_str(seed_str))
    return rng.integers(0, 2, size=dim, dtype=np.uint8)

def encode_token(token: str, dim: int, char_ngram_len: int = 3) -> np.ndarray:
    key = (token, dim)
    if key in _TOKEN_CACHE:
        return _TOKEN_CACHE[key]
        
    # 1. Vecteur d'identité du mot
    word_hv = generate_random_hv(f"word:{token}", dim)
    
    # 2. Si le mot est assez long, on ajoute l'influence des n-grammes de caractères
    if len(token) >= char_ngram_len:
        char_hvs = []
        for i in range(len(token) - char_ngram_len + 1):
            ngram = token[i:i + char_ngram_len]
            ngram_hv = generate_random_hv(f"char:{ngram}", dim)
            char_hvs.append(np.roll(ngram_hv, i))
        
        if char_hvs:
            sum_hvs = np.sum(char_hvs, axis=0, dtype=np.int32)
            char_bundle = (sum_hvs > len(char_hvs) // 2).astype(np.uint8)
            res = np.bitwise_xor(word_hv, char_bundle)
            _TOKEN_CACHE[key] = res
            return res
            
    _TOKEN_CACHE[key] = word_hv
    return word_hv

def rotate(hv: np.ndarray, pos: int) -> np.ndarray:
    shift = PRIMES[pos % len(PRIMES)]
    return np.roll(hv, shift)

def encode_context(tokens: list[str], dim: int) -> np.ndarray:
    if not tokens:
        return np.zeros(dim, dtype=np.uint8)
    
    result = np.zeros(dim, dtype=np.uint8)
    for i, token in enumerate(reversed(tokens)):
        hv = encode_token(token, dim)
        hv_pos = rotate(hv, i)
        result = np.bitwise_xor(result, hv_pos)
    return result

def hamming(hv1: np.ndarray, hv2: np.ndarray) -> int:
    return int(np.count_nonzero(np.bitwise_xor(hv1, hv2)))
