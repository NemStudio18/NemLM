"""
HDC-LLM V2 - Moteur de Q&A (Multi-Layer Ranking)
"""

import argparse
import sys
import os
import time
from hdc.semantic import SemanticIndex

def split_into_sentences(text: str) -> list[str]:
    text = text.replace("\n", " ").replace("\r", "")
    import re
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="Document à indexer")
    parser.add_argument("--dim", type=int, default=30000)
    args = parser.parse_args()

    print(f"\n=== HDC Q&A Engine V2.1 (D={args.dim}) ===")
    print("      Multi-Layer Ranking (BoW + Sequence)")
    
    index = SemanticIndex(dim=args.dim, lsh_bits=12)

    if args.file and os.path.exists(args.file):
        with open(args.file, "r", encoding="utf-8") as f:
            content = f.read()
        sentences = split_into_sentences(content)
        print(f"[1/2] Indexation de {len(sentences)} phrases...")
        index.build(sentences)
    else:
        test_doc = [
            "L'intelligence artificielle binaire est une approche qui n'utilise pas de nombres flottants.",
            "Elle fonctionne sur des processeurs anciens comme l'Intel i3-3220.",
            "Le protocole NHTML permet de piloter le DOM avec des patchs binaires.",
            "L'HDC utilise des vecteurs de haute dimension pour représenter la connaissance.",
            "La latence est très basse car on utilise des opérations XOR et popcount.",
            "Le FX-8350 possède 28 Go de RAM DDR3.",
            "Rust est utilisé pour les parties critiques afin d'atteindre une latence sub-100 microsecondes."
        ]
        print("[1/2] Utilisation du document de test par défaut...")
        index.build(test_doc)

    print("[2/2] Moteur prêt. Posez vos questions.\n")

    while True:
        try:
            query = input("Question > ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if query.lower() in ("exit", "quit", "q"):
            break
        if not query:
            continue

        t0 = time.perf_counter()
        # On demande le Top-3 avec re-ranking sur le Top-20 initial
        results = index.query(query, k=3, rerank_top_n=20)
        t1 = time.perf_counter()
        
        elapsed_ms = (t1 - t0) * 1000

        print(f"\n  Résultats ({elapsed_ms:.2f} ms) :")
        for i, (text, score, confidence) in enumerate(results, 1):
            print(f"    {i}. [{confidence:.1f}%] {text}")
        print()

if __name__ == "__main__":
    main()
