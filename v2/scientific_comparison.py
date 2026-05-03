import time
import os
import random
import sys
import io
from hdc.v3_engine  import V3Engine
from hdc.corpus     import load, build_vocabulary
from eval_kneser_ney import KneserNeyModel

# Force UTF-8 pour la console Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def run_benchmark():
    print("=== BENCHMARK SCIENTIFIQUE : NemLM vs KNESER-NEY ===")
    
    # Parametres
    train_limit = 25000
    test_size   = 500
    dim         = 10000
    
    path = "europarl_fr.txt"
    if not os.path.exists(path):
        print("Erreur: corpus europarl_fr.txt introuvable.")
        return

    # Chargement
    print(f"Chargement des données...")
    all_sents = load(path)
    train_sents = all_sents[:train_limit]
    test_sents  = all_sents[train_limit:train_limit + test_size]
    
    vocab = build_vocabulary(train_sents, max_size=30000)
    print(f"Vocabulaire : {len(vocab):,} mots")
    print(f"Entrainement sur {len(train_sents):,} phrases.")

    # 1. Entrainement Kneser-Ney
    kn_model = KneserNeyModel(n=5)
    t0 = time.perf_counter()
    kn_model.train(train_sents)
    kn_time = time.perf_counter() - t0
    print(f"Kneser-Ney entrainé en {kn_time:.2f}s")

    # 2. Entrainement NemLM (HDC)
    hdc_engine = V3Engine(dim=dim)
    t0 = time.perf_counter()
    for sent in train_sents:
        hdc_engine.train_step(" ".join(sent))
    hdc_engine.rebuild_lsh(vocab)
    hdc_time = time.perf_counter() - t0
    print(f"NemLM (HDC) entrainé en {hdc_time:.2f}s")

    # 3. ÉVALUATION (Cloze Test)
    print(f"\nÉvaluation sur {test_size} questions...")
    hdc_hits = 0
    kn_hits  = 0
    
    for sent in test_sents:
        if len(sent) < 5: continue
        context = sent[:4]
        target  = sent[4]
        
        # Predict Top-5
        combined_hv = hdc_engine.get_combined_hv(context)
        hdc_preds   = hdc_engine.memory.predict_topk(combined_hv, k=5)
        kn_preds    = kn_model.predict_topk(context, k=5)
        
        if target in hdc_preds: hdc_hits += 1
        if target in kn_preds: kn_hits += 1

    print("\n" + "="*40)
    print("RÉSULTATS FINAUX (Top-5 Accuracy)")
    print("="*40)
    print(f"NemLM (HDC) : {hdc_hits/test_size*100:>6.2f}%")
    print(f"Kneser-Ney  : {kn_hits/test_size*100:>6.2f}%")
    print("-" * 40)
    print(f"Ratio Vitesse Entrainement : {hdc_time/kn_time:.1f}x")
    print("="*40)

if __name__ == "__main__":
    run_benchmark()
