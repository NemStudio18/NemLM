import time
import os
import gc
import numpy as np
from hdc.v3_engine import V3Engine, encode_context
from hdc.corpus import load, tokenize
from hdc.representation import DIM
from eval_kneser_ney import KneserNeyModel

def format_time(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}m {s:02d}s"

def run_duel_v5():
    print("="*60)
    print("      NEMLM V5 - DUEL PERSISTANT (HYBRID ATTENTION)")
    print("="*60)
    
    sentences = load("../europarl_fr.txt", max_tokens=2000000)
    train_sents = sentences[:25000]
    test_sents = sentences[25000:30000]

    db_path = r"D:\nemlm_v5_stable.nemdb"
    engine = V3Engine(DIM, db_path=db_path)
    kn_model = KneserNeyModel(n=5)
    
    print(f"[*] Base de connaissances : {db_path}")
    print("[*] Entrainement Kneser-Ney 5-grammes...")
    kn_model.train(train_sents)

    resume_at = 25000
    print(f"\n[>>>] RATTRAPAGE ATTENTION RAM ({resume_at} phrases)")
    start_train = time.time()
    
    for i, sent in enumerate(train_sents):
        if len(sent) < 2: continue
        for j in range(1, len(sent)):
            context = sent[max(0, j-5):j]
            target = sent[j]
            l_hv_packed = encode_context(context, engine.dim)
            target_hv = engine.semantic.get_word_hv(target)
            engine.attention.learn(l_hv_packed, target_hv)
        
        if (i + 1) % 1000 == 0:
            elapsed = time.time() - start_train
            speed = (i + 1) / (elapsed + 0.1)
            eta = (resume_at - (i + 1)) / speed
            print(f" > [{format_time(elapsed)}] {i+1:5d}/{resume_at} | Speed: {speed:5.1f} phr/s | ETA: {format_time(eta)}")
    
    print(f"\n[OK] Attention restaurée en {format_time(time.time() - start_train)}")
    print("[*] Sauvegarde de l'Attention sur disque...")
    engine.commit() 
    
    # --- EVALUATION ---
    print(f"\n[>>>] ÉVALUATION FINALE (5000 questions)")
    hdc_hits = 0
    kn_hits = 0
    total = 0
    
    start_eval = time.time()
    for i, sent in enumerate(test_sents):
        if len(sent) < 5: continue
        context = sent[:4]
        target = sent[4]
        
        # NemLM Predict (Top-5)
        preds_hdc = engine.predict_next(context, top_k=5)
        if target in preds_hdc: hdc_hits += 1
        
        # KN Predict (Top-5)
        preds_kn = kn_model.predict_topk(context, k=5)
        if target in preds_kn: kn_hits += 1
        
        total += 1
        if total % 100 == 0:
            elapsed = time.time() - start_eval
            speed = total / (elapsed + 0.1)
            eta = (len(test_sents) - total) / speed
            print(f" > [{format_time(elapsed)}] Evalué {total:4d}/5000 | NemLM: {hdc_hits/total*100:5.2f}% | KN: {kn_hits/total*100:5.2f}% | ETA: {format_time(eta)}")

    duration = time.time() - start_eval
    
    print("\n" + "="*60)
    print("                RÉSULTATS FINAUX - DUEL V5")
    print("="*60)
    print(f" NemLM (Hybrid V5) : {hdc_hits/total*100:.2f}% (Top-5 Accuracy)")
    print(f" Kneser-Ney (Ref)  : {kn_hits/total*100:.2f}% (Top-5 Accuracy)")
    print(f" Temps Total Eval  : {format_time(duration)}")
    print(f" Status            : Modèle Persistant Sauvegardé sur D:")
    print("="*60)

if __name__ == "__main__":
    run_duel_v5()
