"""
HDC Representation Layer
Zéro flottant. Zéro GPU. Vecteurs binaires purs.
"""

import hashlib
import numpy as np

DIM = 10000  # Dimension hypervectorielle


def _seed_from_token(token: str) -> int:
    """Seed reproductible et unique par token via SHA256."""
    h = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big")


def encode(token: str, dim: int = DIM) -> np.ndarray:
    """
    Token → HyperVector binaire (dtype uint8, valeurs 0/1).
    Reproductible : même token → même vecteur.
    """
    rng = np.random.default_rng(_seed_from_token(token))
    return rng.integers(0, 2, size=dim, dtype=np.uint8)


def rotate(hv: np.ndarray, n: int) -> np.ndarray:
    """Rotation circulaire de n bits — encode la position dans le contexte."""
    return np.roll(hv, n)


def xor(hv1: np.ndarray, hv2: np.ndarray) -> np.ndarray:
    """XOR binaire — combine deux hypervecteurs."""
    return np.bitwise_xor(hv1, hv2)


def encode_context(tokens: list[str], dim: int = DIM) -> np.ndarray:
    """
    Liste de tokens → HyperVector contexte.
    Chaque token est rotaté selon sa position pour préserver l'ordre.
    Résultat = XOR de tous les tokens positionnés.
    """
    if not tokens:
        return np.zeros(dim, dtype=np.uint8)

    result = np.zeros(dim, dtype=np.uint8)
    for i, token in enumerate(tokens):
        hv = encode(token, dim)
        hv_pos = rotate(hv, i)
        result = xor(result, hv_pos)
    return result


def hamming(hv1: np.ndarray, hv2: np.ndarray) -> int:
    """
    Distance de Hamming = nombre de bits différents.
    Opération CPU native : XOR + popcount.
    Zéro flottant.
    """
    return int(np.count_nonzero(np.bitwise_xor(hv1, hv2)))


def similarity(hv1: np.ndarray, hv2: np.ndarray) -> int:
    """
    Similarité = DIM - hamming.
    Plus grand = plus similaire.
    """
    return hv1.shape[0] - hamming(hv1, hv2)
