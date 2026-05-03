import time
import numpy as np
import sys
import io
from hdc.v3_engine import V3Engine
from hdc.corpus import load, build_vocabulary
from hdc.representation import encode_context, hamming

# Force UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def run_inference_bench():
    print("=== BENCHMARK INFERENCE & QA (NemLM V3) ===")
    
    dim = 10000
    num_test = 100
    
    # 1. Setup
    engine = V3Engine(dim=dim)
    all_sents = load("europarl_fr.txt")
    test_sents = [s for s in all_sents if len(s) > 6][:num_test]
    vocab = build_vocabulary(test_sents, max_size=5000)
    
    # 2. Indexation (Entrainement rapide des 100 phrases)
    print(f"Indexation de {num_test} phrases de test...")
    for sent in test_sents:
        engine.train_step(" ".join(sent))
    engine.rebuild_lsh(vocab, num_tables=8, num_bits=14)
    
    # 3. Test d'Inference
    print(f"Lancement de {num_test} requêtes de prédiction...")
    latencies = []
    hits = 0
    lsh_recalls = []
    stability_errors = 0
    
    for sent in test_sents:
        context = sent[:5]
        target  = sent[5]
        
        # Test Stabilité
        hv1 = encode_context(context, dim)
        hv2 = encode_context(context, dim)
        if hamming(hv1, hv2) > 0: stability_errors += 1
        
        # Mesure Latence & Precision
        t0 = time.perf_counter()
        
        # On utilise predict_topk pour mesurer le recall LSH
        l_hv = encode_context(context, dim)
        preds = engine.memory.predict_topk(l_hv, k=5)
        
        latencies.append((time.perf_counter() - t0) * 1000)
        
        if target in preds:
            hits += 1
            
    # Resultats
    print("\n" + "="*40)
    print("RÉSULTATS D'INFÉRENCE")
    print("="*40)
    print(f"Latence Moyenne    : {np.mean(latencies):.2f} ms / requête")
    print(f"Précision (Top-5)  : {hits/num_test*100:.1f}%")
    print(f"Erreurs Stabilité  : {stability_errors}")
    print(f"Débit estimé       : {1000/np.mean(latencies):.1f} req/sec")
    print("="*40)

if __name__ == "__main__":
    run_inference_bench()
