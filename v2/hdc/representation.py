"""
HDC Representation Layer V3 (Bit-Packed)
- Stockage compact par np.packbits.
- weighted_sum en int8 pour economie RAM.
- Hamming optimise sur bits packes.
"""
import hashlib
import numpy as np
from functools import lru_cache

DIM = 10000
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]

# Cache pour les vecteurs de tokens (STOCKAGE PACKE)
_TOKEN_CACHE = {}

def _seed_from_str(s: str) -> int:
    h = hashlib.sha256(s.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big")

def generate_random_hv(seed_str: str, dim: int) -> np.ndarray:
    rng = np.random.default_rng(_seed_from_str(seed_str))
    # Genere et pack immediatement
    bits = rng.integers(0, 2, size=dim, dtype=np.uint8)
    return np.packbits(bits)

def encode_token(token: str, dim: int, char_ngram_len: int = 3) -> np.ndarray:
    """Retourne un vecteur PACKE (dim/8 octets)."""
    key = (token, dim)
    if key in _TOKEN_CACHE:
        return _TOKEN_CACHE[key]
        
    # Genere en mode deballe pour les calculs internes
    rng = np.random.default_rng(_seed_from_str(f"word:{token}"))
    word_bits = rng.integers(0, 2, size=dim, dtype=np.uint8)
    
    if len(token) >= char_ngram_len:
        char_hvs = []
        for i in range(len(token) - char_ngram_len + 1):
            ngram = token[i:i + char_ngram_len]
            # Pour la rotation, on a besoin des bits reels
            ngram_rng = np.random.default_rng(_seed_from_str(f"char:{ngram}"))
            ngram_bits = ngram_rng.integers(0, 2, size=dim, dtype=np.uint8)
            char_hvs.append(np.roll(ngram_bits, i))
        
        if char_hvs:
            sum_hvs = np.sum(char_hvs, axis=0, dtype=np.int16)
            char_bundle = (sum_hvs > len(char_hvs) // 2).astype(np.uint8)
            word_bits = np.bitwise_xor(word_bits, char_bundle)
            
    packed = np.packbits(word_bits)
    _TOKEN_CACHE[key] = packed
    return packed

@lru_cache(maxsize=10000)
def rotate(hv_packed_bytes: bytes, pos: int, dim: int = DIM) -> np.ndarray:
    """Rotation bit-level sur vecteur packe (entree en bytes pour le cache)."""
    hv_packed = np.frombuffer(hv_packed_bytes, dtype=np.uint8)
    bits = np.unpackbits(hv_packed)[:dim]
    shift = PRIMES[pos % len(PRIMES)]
    rotated = np.roll(bits, shift)
    return np.packbits(rotated)

def encode_context(tokens: list[str], dim: int) -> np.ndarray:
    """Retourne un vecteur PACKE."""
    packed_result = np.zeros(dim // 8 + (1 if dim % 8 != 0 else 0), dtype=np.uint8)
    if not tokens: return packed_result
    
    for i, token in enumerate(reversed(tokens)):
        hv_packed = encode_token(token, dim)
        # On passe en bytes pour que le lru_cache fonctionne (np.ndarray n'est pas hashable)
        hv_pos_packed = rotate(hv_packed.tobytes(), i, dim)
        packed_result = np.bitwise_xor(packed_result, hv_pos_packed)
    return packed_result

def hamming(hv1_packed: np.ndarray, hv2_packed: np.ndarray) -> int:
    """Hamming ultra-rapide sur vecteurs packes."""
    xor_res = np.bitwise_xor(hv1_packed, hv2_packed)
    # unpackbits sur le resultat du XOR donne directement les bits a 1
    return int(np.unpackbits(xor_res).sum())
