"""
NemLM Quick-Check V3c
- prewarm_cache UNE FOIS au demarrage (~50s)
- LSH cles cachees : rebuild = ~75ms au lieu de ~11s
- train_step sans predict : ~1.5s/500 phrases
- TOTAL estime par palier : 2-4s (vs 90s avant)
"""

import time
import sys
import os

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from hdc.v3_engine  import V3Engine
from hdc.corpus     import load, build_vocabulary
from hdc.representation import encode_token


def prewarm(vocab: list[str], dim: int) -> None:
    """Prechauffe le cache encode_token pour tout le vocabulaire."""
    n = len(vocab)
    print(f"Prechauffage du cache ({n:,} tokens)...", end="", flush=True)
    t0 = time.perf_counter()
    for w in vocab:
        encode_token(w, dim)
    print(f" OK en {time.perf_counter()-t0:.1f}s")


def progress_bar(current, total, bar_length=20):
    fraction = max(0.0, min(1.0, current / total))
    arrow    = int(fraction * bar_length) * "#"
    padding  = (bar_length - len(arrow)) * "."
    return f"[{arrow}{padding}] {int(fraction * 100):3}%"


def main():
    print("DEMARRAGE DU QUICK-CHECK V3c...")

    dim   = 10000
    step  = 500
    limit = 10000
    paliers = list(range(step, limit + step, step))

    # Corpus
    base  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path  = os.path.join(base, "legacy", "text8")
    all_s = load(path)
    sents = all_s[:limit]
    vocab = build_vocabulary(sents)
    print(f"Vocabulaire ({limit//1000}k) : {len(vocab):,} mots")

    # Prewarm UNE FOIS (encode_token froid = 0.6ms/tok, chaud = 0.0002ms)
    prewarm(vocab, dim)

    # Premier build LSH : calcul des cles (~11s, une seule fois grace au cache)
    print("Construction initiale du LSH (cles cachees pour toujours)...", end="", flush=True)
    engine = V3Engine(dim=dim)
    t0 = time.perf_counter()
    engine.rebuild_lsh(vocab, num_tables=6, num_bits=12)
    print(f" OK en {time.perf_counter()-t0:.1f}s")
    print("Les rebuild() suivants seront <100ms.")

    last_idx = 0

    print("\n" + "=" * 72)
    print(f"{'PHRASES':<10} | {'PROGRESSION':<25} | {'TPS STEP':>10} | GEN TEST")
    print("-" * 72)

    for goal in paliers:
        t0 = time.perf_counter()

        # Apprentissage (sans predict -> O(phrases))
        for i in range(last_idx, goal):
            engine.train_step(" ".join(sents[i]))
            if (i + 1 - last_idx) % 100 == 0:
                p = progress_bar(i + 1 - last_idx, goal - last_idx)
                sys.stdout.write(f"\r{goal:<10} | {p} | Apprentissage... ")
                sys.stdout.flush()

        # Rebuild LSH (75ms grace au cache de cles)
        engine.rebuild_lsh(vocab, num_tables=6, num_bits=12)

        elapsed = time.perf_counter() - t0

        # Generation test
        gen     = engine.generate("the cat", max_new_tokens=4)
        gen_str = " ".join(gen) if gen else "(vide)"

        print(f"\r{goal:<10} | {progress_bar(1,1)} | {elapsed:>8.1f}s | {gen_str}")
        last_idx = goal

    print("\nTermine.")


if __name__ == "__main__":
    main()
