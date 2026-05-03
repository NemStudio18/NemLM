"""
Test de generation longue apres 10k phrases d'entrainement optimise.
Utilise le LSH avec cles cachees et le prewarm.
"""
import time
import sys
import os

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from hdc.v3_engine  import V3Engine
from hdc.corpus     import load, build_vocabulary
from hdc.representation import encode_token


def main():
    dim   = 10000
    limit = 10000

    base   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path   = os.path.join(base, "legacy", "text8")

    print(f"Chargement du corpus...")
    sents = load(path)[:limit]
    vocab = build_vocabulary(sents)
    print(f"Vocabulaire : {len(vocab):,} mots")

    # Prewarm
    print("Prewarm cache...", end="", flush=True)
    t0 = time.perf_counter()
    for w in vocab:
        encode_token(w, dim)
    print(f" {time.perf_counter()-t0:.1f}s")

    engine = V3Engine(dim=dim)

    # Entrainement
    print(f"Entrainement sur {limit:,} phrases...")
    t0 = time.perf_counter()
    for sent in sents:
        engine.train_step(" ".join(sent))
    print(f"Entrainement : {time.perf_counter()-t0:.1f}s")

    # Index LSH final (cles deja cachees depuis prewarm -> rapide)
    print("Construction LSH...", end="", flush=True)
    t0 = time.perf_counter()
    engine.rebuild_lsh(vocab, num_tables=6, num_bits=12)
    print(f" {time.perf_counter()-t0:.1f}s")

    # Generation longue
    prompts = ["the cat", "scientific theory", "history of", "in the year"]

    print("\n" + "=" * 55)
    print("GENERATION LONGUE (30 tokens)")
    print("=" * 55)

    for p in prompts:
        t0  = time.perf_counter()
        gen = engine.generate(p, max_new_tokens=30)
        elapsed = time.perf_counter() - t0
        print(f"\nPROMPT : '{p}'")
        print(f"REPONSE: {' '.join(gen)}")
        print(f"Temps  : {elapsed:.2f}s  ({elapsed/max(len(gen),1)*1000:.0f} ms/token)")

    print("\nTermine.")


if __name__ == "__main__":
    main()
