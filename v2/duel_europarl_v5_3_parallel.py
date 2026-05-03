import time
import os
import numpy as np
import multiprocessing as mp
from hdc.parallel_engine import V3ParallelEngine
from hdc.representation import DIM
from hdc.corpus import load
from eval_kneser_ney import KneserNeyModel

def format_time(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}m {s:02d}s"

def run_duel():
    log_file = "duel_europarl_v5_3_parallel.txt"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("=== NemLM V5.3 PARALLEL vs Kneser-Ney Duel (Europarl FR) ===\n")

    def log_print(msg):
        print(msg, flush=True)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log_print("="*60)
    log_print("      SCIENTIFIC DUEL V5.3 - PARALLEL + PRUNING")
    log_print("="*60)
    
    # Configuration
    TRAIN_SIZE = 15000
    TEST_SIZE = 5000
    DB_PATH = r"D:\nemlm_duel_v5_3.nemdb"
    
    # Nettoyage DB pour un test pur
    if os.path.exists(DB_PATH): 
        try: os.remove(DB_PATH)
        except: pass
    
    # Chargement Corpus
    log_print(f"[*] Chargement Europarl...")
    sentences = load("../europarl_fr.txt", max_tokens=1000000)
    train_sents = sentences[:TRAIN_SIZE]
    test_sents = sentences[TRAIN_SIZE : TRAIN_SIZE + TEST_SIZE]
    
    # Initialisation
    log_print("[*] Initialisation des 3 Workers spécialisés...")
    engine = V3ParallelEngine(DIM, db_path=DB_PATH, num_workers=3)
    kn_model = KneserNeyModel(n=5)
    
    # 1. Entrainement Kneser-Ney
    log_print("[*] Entrainement Kneser-Ney (Ref)...")
    kn_model.train(train_sents)
    
    # 2. Entrainement NemLM V5.3 Parallel
    log_print(f"[*] Entrainement NemLM V5.3 ({TRAIN_SIZE} phrases)...")
    start_train = time.time()
    for i, sent in enumerate(train_sents):
        engine.train_step(sent)
        if (i + 1) % 2500 == 0:
            elapsed = time.time() - start_train
            speed = (i + 1) / (elapsed + 0.1)
            log_print(f"  > Progress: {((i+1)/TRAIN_SIZE)*100:5.1f}% | Speed: {speed:5.1f} phr/s")
    
    log_print("[*] Finalisation des workers (Commit final)...")
    engine.stop()
    log_print(f"[OK] Entrainement NemLM termine en {format_time(time.time() - start_train)}")
    
    # 3. Evaluation
    log_print(f"\n[>>>] EVALUATION FINALE ({TEST_SIZE} questions)")
    hdc_hits = 0
    kn_hits = 0
    total = 0
    
    start_eval = time.time()
    for i, sent in enumerate(test_sents):
        if len(sent) < 5: continue
        context = sent[:4]
        target = sent[4]
        
        # NemLM Predict (Top-5)
        # On utilise predict_next qui ouvre sa propre connexion temporaire
        preds_hdc = engine.predict_next(context, top_k=5)
        if target in preds_hdc: hdc_hits += 1
        
        # KN Predict (Top-5)
        preds_kn = kn_model.predict_topk(context, k=5)
        if target in preds_kn: kn_hits += 1
        
        total += 1
        if total % 500 == 0:
            elapsed = time.time() - start_eval
            speed = total / (elapsed + 0.1)
            log_print(f"  > Evalué {total:4d}/{TEST_SIZE} | NemLM: {hdc_hits/total*100:5.2f}% | KN: {kn_hits/total*100:5.2f}%")

    log_print("\n" + "="*60)
    log_print("                RÉSULTATS FINAUX")
    log_print("="*60)
    log_print(f" NemLM V5.3 (Parallel) : {hdc_hits/total*100:.2f}% (Top-5 Accuracy)")
    log_print(f" Kneser-Ney (Ref)      : {kn_hits/total*100:.2f}% (Top-5 Accuracy)")
    log_print(f" Temps Evaluation      : {format_time(time.time() - start_eval)}")
    log_print("="*60)

if __name__ == "__main__":
    # IMPORTANT: Sous Windows, mp.Process nécessite ce guard
    run_duel()
