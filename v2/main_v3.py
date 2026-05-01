"""
HDC-LLM V3 - Stress Test (500 Phrases / ~25k Tokens)
"""

import time
import os
from hdc.v3_engine import V3Engine
from hdc.corpus import load, build_vocabulary

MODEL_PATH = "model_v3_large.hdb"

def main():
    print("\n" + "="*50)
    print(" HDC-LLM V3 - STRESS TEST (PYTHON) ".center(50, "="))
    print("="*50 + "\n")

    # 1. Chargement du corpus
    print("[1/4] Chargement du corpus (Large subset)...")
    sentences_raw = load("../v1/text8")
    sentences_raw = sentences_raw[:500] 
    vocab = build_vocabulary(sentences_raw)
    
    engine = V3Engine(dim=30000)

    # 2. Entraînement
    print(f"[2/4] Entraînement de {len(sentences_raw)} phrases ({len(vocab)} mots)...")
    t_start = time.perf_counter()
    
    for i, s_tokens in enumerate(sentences_raw):
        sentence_str = " ".join(s_tokens)
        engine.train_step(sentence_str, vocab=vocab)
        
        if (i + 1) % 50 == 0:
            elapsed = time.perf_counter() - t_start
            print(f"      - {i+1}/{len(sentences_raw)} phrases traitées (Time: {elapsed:.1f}s)")
            
    total_time = time.perf_counter() - t_start
    print(f"✅ Entraînement terminé en {total_time:.1f} s")
    print(f"🚀 Débit : {len(sentences_raw)/total_time:.2f} phrases/sec")

    # SAUVEGARDE
    engine.memory.save(MODEL_PATH)

    # 3. Tests de génération (Contexte Stable)
    print("\n[3/4] Génération autorégressive (Inférence)...")
    prompts = [
        "anarchism is a social movement",
        "the working class",
        "ancient history of the"
    ]
    
    for prompt in prompts:
        t0 = time.perf_counter()
        gen = engine.generate(prompt, max_new_tokens=20, vocab=vocab)
        t1 = time.perf_counter()
        print(f"\n  >>> Prompt: '{prompt}' ({ (t1-t0)*1000/len(gen) if gen else 0 :.1f}ms/token)")
        print(f"  Result: '{prompt} {' '.join(gen)}'")

    # 4. Rapport de fichiers
    print(f"\n[4/4] Rapport final :")
    print(f"  - Entrées mémoire : {engine.memory.size}")
    print(f"  - Taille disque : {os.path.getsize(MODEL_PATH)/1024:.1f} KB")

if __name__ == "__main__":
    main()
