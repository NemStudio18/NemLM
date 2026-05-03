"""
HDC Corpus Layer V2
- Support text8.
- Tokenisation robuste.
"""

import re
from pathlib import Path
from typing import Iterator

def tokenize(text: str) -> list[str]:
    text = text.lower().strip()
    # Support des accents francais
    text = re.sub(r"[^a-zàâçéèêëîïôûùÿæœ\s]", " ", text)
    return [t for t in text.split() if t]

def load(path: str, max_tokens: int = None) -> list[list[str]]:
    p = Path(path)
    if not p.exists(): raise FileNotFoundError(path)
    
    with open(p, "r", encoding="utf-8") as f:
        # Pour text8, on lit par morceaux pour ne pas exploser la RAM si c'est 100MB
        content = f.read(max_tokens * 10 if max_tokens else -1)
        
    all_tokens = tokenize(content)
    if max_tokens: all_tokens = all_tokens[:max_tokens]
    
    # Découpe en morceaux de 50 mots
    chunk_size = 50
    return [all_tokens[i:i + chunk_size] for i in range(0, len(all_tokens), chunk_size) if len(all_tokens[i:i + chunk_size]) >= 2]

def build_vocabulary(sentences: list[list[str]], max_size: int = 50000) -> list[str]:
    vocab = {}
    for s in sentences:
        for t in s:
            vocab[t] = vocab.get(t, 0) + 1
    
    # Trie par fréquence et limite la taille
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
    return [item[0] for item in sorted_vocab[:max_size]]

def ngrams(sentences: list[list[str]], n: int) -> Iterator[tuple]:
    for sentence in sentences:
        if len(sentence) < n: continue
        for i in range(len(sentence) - n + 1):
            yield tuple(sentence[i : i + n])

def train_test_split(sentences: list[list[str]], ratio: float = 0.8) -> tuple[list, list]:
    split = int(len(sentences) * ratio)
    return sentences[:split], sentences[split:]
