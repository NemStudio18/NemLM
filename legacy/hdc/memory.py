"""
HDC Memory Layer
Associative memory : contexte -> token suivant.
Zéro flottant. Majority bundling : vote bit par bit, préserve l'information.

Pourquoi majority bundling > XOR bundling :
  XOR de N vecteurs -> les bits s'annulent mutuellement quand N est pair.
  Majority vote -> chaque bit = 1 si la majorité des vecteurs ont 1 à cette position.
  Résultat : le bundle ressemble aux vecteurs fréquents, pas à du bruit.

Stockage interne :
  _counts_store : dict[key -> np.int16 array de dim]
    Accumule le nombre de 1 par position de bit sur tous les tokens suivants vus.
    int16 suffit pour des milliers d'occurrences, zéro flottant.
  _bundle_cache : dict[key -> np.uint8 array de dim]
    Cache du bundle majoritaire (invalidé à chaque learn).

Vectorisation :
  predict_topk calcule toutes les distances Hamming en une seule opération
  matricielle numpy (XOR + sum sur axe), 100x plus rapide que la boucle.
"""

import numpy as np
from .representation import encode, encode_context, DIM


class AssociativeMemory:
    """
    Mémoire associative HDC avec majority bundling et Hamming vectorisé.
    """

    def __init__(self, dim: int = DIM):
        self.dim = dim
        # Accumulation int16 par position de bit : somme des 1 observés
        self._counts_store: dict[int, np.ndarray] = {}
        # Nombre total d'observations par contexte
        self._obs_count: dict[int, int] = {}
        # Cache du bundle binaire (majority vote appliqué)
        self._bundle_cache: dict[int, np.ndarray] = {}
        # Cache des HV de vocabulaire (évite de recalculer à chaque predict)
        self._vocab_matrix: np.ndarray | None = None
        self._vocab_list: list[str] = []

    def _hv_key(self, hv: np.ndarray) -> int:
        """Hash 64 bits d'un HV pour clé de stockage."""
        packed = np.packbits(hv)
        return hash(packed.tobytes()) & 0xFFFFFFFFFFFFFFFF

    def _majority_bundle(self, key: int) -> np.ndarray:
        """
        Calcule le bundle majoritaire depuis l'accumulateur int16.
        Bit i = 1 si counts[i] > total_obs / 2, sinon 0.
        Zéro flottant : comparaison entière pure.
        Cache le résultat jusqu'au prochain learn().
        """
        if key in self._bundle_cache:
            return self._bundle_cache[key]

        counts = self._counts_store[key]
        total = self._obs_count[key]
        threshold = total // 2  # division entière, zéro flottant

        # Comparaison entière : 1 si majorité, 0 sinon
        bundle = (counts > threshold).view(np.uint8)
        self._bundle_cache[key] = bundle
        return bundle

    def learn(self, context_tokens: list[str], next_token: str):
        """
        Apprend l'association contexte -> next_token.
        Accumule les bits du token suivant dans le compteur int16.
        Invalide le cache du bundle pour ce contexte.
        """
        ctx_hv = encode_context(context_tokens, self.dim)
        key = self._hv_key(ctx_hv)
        next_hv = encode(next_token, self.dim)

        if key not in self._counts_store:
            self._counts_store[key] = np.zeros(self.dim, dtype=np.int16)
            self._obs_count[key] = 0

        # Accumulation entière des bits
        self._counts_store[key] += next_hv.astype(np.int16)
        self._obs_count[key] += 1

        # Invalide le cache bundle pour ce contexte
        self._bundle_cache.pop(key, None)

        # Invalide le cache vocab matrix (vocabulaire peut changer)
        self._vocab_matrix = None

    def _build_vocab_matrix(self, vocabulary: list[str]) -> np.ndarray:
        """
        Construit une matrice (vocab_size x dim) des HV de tous les tokens.
        Permet le calcul vectorisé de toutes les distances en une passe.
        Mis en cache tant que le vocabulaire ne change pas.
        """
        if self._vocab_list == vocabulary and self._vocab_matrix is not None:
            return self._vocab_matrix

        matrix = np.stack([encode(t, self.dim) for t in vocabulary])
        self._vocab_matrix = matrix
        self._vocab_list = vocabulary
        return matrix

    def predict_topk(
        self, context_tokens: list[str], vocabulary: list[str], k: int = 3
    ) -> list[tuple[str, int]]:
        """
        Retourne les k tokens les plus probables avec leur distance Hamming.
        Vectorisé : une seule opération matricielle pour tout le vocabulaire.
        Zéro flottant : XOR uint8 + sum int.
        """
        ctx_hv = encode_context(context_tokens, self.dim)
        key = self._hv_key(ctx_hv)

        if key not in self._counts_store:
            bundle = ctx_hv
        else:
            bundle = self._majority_bundle(key)

        # Matrice vocab (V x dim) XOR bundle (dim,) -> distances (V,) en une passe
        vocab_matrix = self._build_vocab_matrix(vocabulary)
        distances = np.count_nonzero(
            np.bitwise_xor(vocab_matrix, bundle[np.newaxis, :]), axis=1
        )

        # Top-k par distance croissante
        top_k_idx = np.argpartition(distances, min(k, len(vocabulary) - 1))[:k]
        top_k_idx = top_k_idx[np.argsort(distances[top_k_idx])]

        return [(vocabulary[i], int(distances[i])) for i in top_k_idx]

    def predict(self, context_tokens: list[str], vocabulary: list[str]) -> str:
        results = self.predict_topk(context_tokens, vocabulary, k=1)
        return results[0][0] if results else ""

    @property
    def size(self) -> int:
        return len(self._counts_store)

    @property
    def memory_bytes(self) -> int:
        if not self._counts_store:
            return 0
        # int16 = 2 bytes par position, dim positions
        bytes_per_ctx = self.dim * 2
        return self.size * bytes_per_ctx
