"""
HDC-LLM V2 - Point d'entrée principal
Supporte D=100k, LSH (K=16), Passive-Aggressive, text8.
"""

import argparse
import sys
import os
from hdc.corpus import load, build_vocabulary, train_test_split, ngrams
from hdc.memory import AssociativeMemory
from hdc.eval import evaluate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="../v1/text8", help="Chemin du corpus")
    parser.add_argument("--dim", type=int, default=30000, help="Dimension (30k-100k)")
    parser.add_argument("--context", type=int, default=5, help="N-gramme (1-9)")
    parser.add_argument("--epochs", type=int, default=3, help="Époques Perceptron")
    parser.add_argument("--max-tokens", type=int, default=100000, help="Limiter la taille du corpus")
    parser.add_argument("--no-lsh", action="store_true", help="Désactiver LSH")
    args = parser.parse_args()

    print(f"\n=== HDC-LLM V2.0 (D={args.dim}, N={args.context}) ===")
    
    # 1. Chargement
    print(f"[1/5] Chargement du corpus ({args.max_tokens} tokens)...")
    if not os.path.exists(args.corpus):
        print(f"Erreur: Corpus non trouvé à {args.corpus}")
        return
        
    sentences = load(args.corpus, max_tokens=args.max_tokens)
    vocab = build_vocabulary(sentences)
    train_set, test_set = train_test_split(sentences)
    
    print(f"      Vocabulaire : {len(vocab)} tokens")
    print(f"      Train/Test  : {len(train_set)}/{len(test_set)} blocs")

    # 2. Mémoire
    memory = AssociativeMemory(dim=args.dim, use_lsh=not args.no_lsh)

    # 3. Phase 1 : Bundling
    print("[2/5] Phase 1 : Apprentissage par Bundling...")
    for sentence in train_set:
        for gram in ngrams([sentence], args.context + 1):
            memory.learn_one_pass(list(gram[:-1]), gram[-1])
    print(f"      Contextes mémorisés : {memory.size}")

    # 4. Phase 2 : LSH
    if not args.no_lsh:
        print("[3/5] Phase 2 : Construction de l'index LSH (L=10, K=16)...")
        memory.build_lsh(vocab, num_tables=10, num_bits=16)

    # 5. Phase 3 : Passive-Aggressive Correction
    if args.epochs > 0:
        print(f"[4/5] Phase 3 : Correction Passive-Aggressive ({args.epochs} époques)...")
        for epoch in range(args.epochs):
            errors = 0
            count = 0
            for sentence in train_set:
                for gram in ngrams([sentence], args.context + 1):
                    ctx = list(gram[:-1])
                    target = gram[-1]
                    preds = memory.predict_topk(ctx, vocab, k=1)
                    pred = preds[0][0] if preds else ""
                    
                    if pred != target:
                        memory.update_passive_aggressive(ctx, target, pred, margin=args.dim // 20)
                        errors += 1
                    count += 1
            print(f"      Époque {epoch+1}/{args.epochs} - Erreurs: {errors}/{count} ({errors/count*100:.1f}%)")

    # 6. Évaluation
    print("[5/5] Évaluation finale...")
    evaluate(memory, test_set, vocab, context_size=args.context)

if __name__ == "__main__":
    main()
