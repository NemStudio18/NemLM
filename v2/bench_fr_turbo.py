"""
NemLM V3 - Benchmark Francais Turbo (Bit-Packed & Dual Context)
Test de coherence avec zero pruning sur 10k phrases.
"""
import time
import os
import random
import sys
import io
from hdc.v3_engine import V3Engine
from hdc.corpus import load, build_vocabulary

# Force UTF-8 pour la console Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def main():
    print("=== BENCHMARK FRANCAIS (MODE SECURISE 16GB) ===")

    dim         = 10000
    phrase_limit = 10000
    vocab_limit  = 20000 

    path = "europarl_fr.txt"
    if not os.path.exists(path):
        print("Erreur: corpus europarl_fr.txt introuvable.")
        return

    # Chargement
    print(f"Chargement de {phrase_limit} phrases...")
    sents = load(path)[:phrase_limit]
    vocab = build_vocabulary(sents, max_size=vocab_limit)
    print(f"Vocabulaire : {len(vocab):,} mots")

    engine = V3Engine(dim=dim)
    t_total_start = time.perf_counter()
    
    # Echantillons pour le Recall LSH
    training_samples = []

    print("\nEntrainement (Double Context : Local + Global)...")
    for i, sent in enumerate(sents):
        text = " ".join(sent)
        engine.train_step(text)
        
        # Collecte d'echantillons pour le Recall (Local context)
        if i % 100 == 0 and len(sent) > 5:
            from hdc.representation import encode_context
            ctx_packed = encode_context(sent[:5], dim)
            target = sent[5] if len(sent) > 5 else sent[-1]
            training_samples.append((ctx_packed, target))

    # Indexation LSH (K=14 pour gerer la densite de 500k entrees)
    print("\nReconstruction LSH (K=14)...")
    engine.rebuild_lsh(vocab, num_tables=8, num_bits=14)
    
    # Mesure Recall LSH
    print("Mesure du Recall LSH sur 100 contextes...")
    hits = 0
    test_samples = random.sample(training_samples, min(100, len(training_samples)))
    for ctx_packed, target in test_samples:
        # predict_topk sur l_hv_packed (contexte local)
        preds = engine.memory.predict_topk(ctx_packed, k=5)
        if target in preds: hits += 1
    print(f"LSH Recall: {hits}%")

    # Generation du rapport final
    print("\nGeneration du rapport de specialisation...")
    prompts = ["monsieur le président", "la commission européenne", "le projet de loi"]
    
    with open("v2/rapport_francais.md", "w", encoding="utf-8") as f:
        f.write("# 🇫🇷 Rapport NemLM - Specialisation Francais (Honnête)\n\n")
        f.write(f"- Phrases : {len(sents):,}\n")
        f.write(f"- Vocab : {len(vocab):,}\n")
        f.write(f"- LSH Recall : {hits}%\n")
        f.write(f"- Temps : {time.perf_counter()-t_total_start:.1f}s\n\n")
        f.write("## Exemples de 30 mots :\n")
        for p in prompts:
            g = engine.generate(p, max_new_tokens=30)
            f.write(f"**Prompt** : `{p}`\n")
            f.write(f"**Reponse**: {' '.join(g)}\n\n")
            print(f"Prompt: {p} -> {' '.join(g)[:60]}...")

    print("\nTERMINE. Rapport : v2/rapport_francais.md")

if __name__ == "__main__":
    main()
