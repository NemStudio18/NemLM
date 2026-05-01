"""
HDC-LLM POC - Point d'entrée principal
Prédicteur de token CPU-natif, zéro flottant, zéro GPU.

Usage:
  python main.py --train --eval
  python main.py --interactive
  python main.py --train --eval --interactive
"""

import argparse
import sys
from hdc import (
    AssociativeMemory,
    load, build_vocabulary, ngrams, train_test_split,
    evaluate, save_results,
    DIM,
)


def train(memory: AssociativeMemory, sentences: list, vocabulary: list, context_size: int = 2):
    """Entraîne la mémoire HDC sur le corpus."""
    n = context_size + 1
    count = 0
    for gram in ngrams(sentences, n):
        context = list(gram[:-1])
        next_token = gram[-1]
        memory.learn(context, next_token)
        count += 1
    print(f"  {count} associations apprises ({memory.size} contextes uniques)")


def interactive_mode(memory: AssociativeMemory, vocabulary: list, context_size: int = 2):
    """Mode interactif : saisie de contexte -> prédiction top 3."""
    print("\n-- Mode interactif -------------------------")
    print(f"  Entrez {context_size} mot(s) séparés par des espaces.")
    print("  'quit' pour quitter.\n")

    while True:
        try:
            raw = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nAu revoir.")
            break

        if raw in ("quit", "exit", "q"):
            break

        if not raw:
            continue

        tokens = raw.split()

        if len(tokens) < context_size:
            print(f"  ! Entrez au moins {context_size} mot(s).")
            continue

        # Prend les context_size derniers tokens
        context = tokens[-context_size:]

        import time
        t0 = time.perf_counter_ns()
        results = memory.predict_topk(context, vocabulary, k=3)
        t1 = time.perf_counter_ns()
        elapsed_us = (t1 - t0) / 1000

        print(f"  Contexte : {' '.join(context)}")
        print(f"  Top 3 prédictions :")
        for i, (token, dist) in enumerate(results, 1):
            bar = "#" * (20 - dist // 500)  # visualisation relative
            print(f"    {i}. {token:<15} dist={dist}  {bar}")
        print(f"  Temps : {elapsed_us:.1f} us\n")


def main():
    parser = argparse.ArgumentParser(description="HDC-LLM POC - CPU natif, zéro flottant")
    parser.add_argument("--corpus", default="corpus.txt", help="Chemin du corpus")
    parser.add_argument("--dim", type=int, default=DIM, help="Dimension HDC (défaut: 10000)")
    parser.add_argument("--context", type=int, default=2, help="Taille du contexte (défaut: 2)")
    parser.add_argument("--train", action="store_true", help="Entraîner la mémoire")
    parser.add_argument("--eval", action="store_true", help="Évaluer les performances")
    parser.add_argument("--interactive", action="store_true", help="Mode interactif")
    parser.add_argument("--save", action="store_true", help="Sauvegarder les résultats")
    args = parser.parse_args()

    if not any([args.train, args.eval, args.interactive]):
        parser.print_help()
        sys.exit(0)

    # Chargement corpus
    print(f"\n-- HDC-LLM POC -----------------------------")
    print(f"  Dimension HDC : {args.dim} bits")
    print(f"  Taille contexte : {args.context} tokens")
    print(f"  Corpus : {args.corpus}")

    sentences = load(args.corpus)
    vocabulary = build_vocabulary(sentences)
    train_sentences, test_sentences = train_test_split(sentences, ratio=0.8)

    print(f"  Phrases : {len(sentences)} ({len(train_sentences)} train / {len(test_sentences)} test)")
    print(f"  Vocabulaire : {len(vocabulary)} tokens")
    print(f"  Flottants : 0 OK")
    print("--------------------------------------------")

    # Mémoire HDC
    memory = AssociativeMemory(dim=args.dim)

    # Entraînement
    if args.train or args.eval or args.interactive:
        print("\n[1/3] Entraînement...")
        train(memory, train_sentences, vocabulary, context_size=args.context)

    # Évaluation
    if args.eval:
        print("\n[2/3] Évaluation sur train...")
        train_metrics = evaluate(memory, train_sentences, vocabulary, context_size=args.context, verbose=True)
        train_metrics["split"] = "train"

        print("[2/3] Évaluation sur test...")
        test_metrics = evaluate(memory, test_sentences, vocabulary, context_size=args.context, verbose=True)
        test_metrics["split"] = "test"

        if args.save:
            save_results({"train": train_metrics, "test": test_metrics})

    # Mode interactif
    if args.interactive:
        print("\n[3/3] Mode interactif...")
        interactive_mode(memory, vocabulary, context_size=args.context)


if __name__ == "__main__":
    main()
