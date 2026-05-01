"""
HDC Corpus Layer
Chargement, tokenisation, génération de n-grammes.
"""

import re
from pathlib import Path
from typing import Iterator


def tokenize(sentence: str) -> list[str]:
    """
    Tokenisation simple : lowercase + split sur espaces et ponctuation.
    Pas de stemming. Garde les apostrophes contractées comme un token.
    """
    sentence = sentence.lower().strip()
    # Sépare ponctuation sauf apostrophe interne (l'chat → l' chat)
    sentence = re.sub(r"([.,!?;:«»\(\)])", r" \1 ", sentence)
    tokens = [t for t in sentence.split() if t]
    return tokens


def load(path: str) -> list[list[str]]:
    """
    Charge un corpus texte.
    Gère les fichiers avec phrases par ligne ET les fichiers massifs une seule ligne (text8).
    Retourne une liste de phrases tokenisées.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Corpus introuvable : {path}")

    with open(p, "r", encoding="utf-8") as f:
        content = f.read()

    # Si le fichier est très long sans retours à la ligne (ex: text8)
    if "\n" not in content[:1000] and len(content) > 1000:
        all_tokens = tokenize(content)
        # Découpe en "phrases" de 50 mots pour l'entraînement n-gramme
        chunk_size = 50
        sentences = [all_tokens[i:i + chunk_size] for i in range(0, len(all_tokens), chunk_size)]
        return [s for s in sentences if len(s) >= 2]

    # Sinon, traitement classique par ligne
    sentences = []
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        tokens = tokenize(line)
        if len(tokens) >= 2:
            sentences.append(tokens)
    return sentences


def build_vocabulary(sentences: list[list[str]]) -> list[str]:
    """Construit le vocabulaire unique du corpus, trié."""
    vocab = set()
    for sentence in sentences:
        vocab.update(sentence)
    return sorted(vocab)


def ngrams(sentences: list[list[str]], n: int) -> Iterator[tuple]:
    """
    Génère tous les n-grammes du corpus.
    Pour n=3 : (t0, t1, t2) où t2 est le token à prédire depuis (t0, t1).
    """
    for sentence in sentences:
        if len(sentence) < n:
            continue
        for i in range(len(sentence) - n + 1):
            yield tuple(sentence[i : i + n])


def train_test_split(
    sentences: list[list[str]], ratio: float = 0.8
) -> tuple[list, list]:
    """Split 80/20 déterministe (pas de shuffle pour reproductibilité)."""
    split = int(len(sentences) * ratio)
    return sentences[:split], sentences[split:]
