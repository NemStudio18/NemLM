import time
import os
import numpy as np
from hdc.v3_engine import V3Engine

def load_europarl(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        sentences = f.readlines()
    return [s.strip().lower() for s in sentences]

def evaluate_full(db_path, test_data, output_file):
    print(f"[*] Evaluation du moteur FULL : {db_path}")
    engine = V3Engine(db_path=db_path)
    
    top1_correct = 0
    top5_correct = 0
    total_questions = 0
    
    start_time = time.time()
    
    for sentence in test_data:
        tokens = sentence.split()
        if len(tokens) < 2: continue
        
        # Reset de l'accumulateur pour chaque phrase (Teacher Forcing local)
        engine.accumulator.reset()
        
        for i in range(1, len(tokens)):
            context = tokens[max(0, i - 5):i]
            target = tokens[i]
            
            # Prediction
            preds = engine.predict_next(context, top_k=5)
            
            if preds:
                if preds[0] == target:
                    top1_correct += 1
                if target in preds:
                    top5_correct += 1
            
            # IMPORTANT: Teacher Forcing (injection de la vraie target pour le contexte suivant)
            target_hv = engine.semantic.get_word_hv(target)
            engine.accumulator.add(target_hv)
            
            total_questions += 1
            if total_questions % 500 == 0:
                cur_acc5 = (top5_correct / total_questions) * 100
                print(f"[*] FULL Progrès : {total_questions} questions | Accuracy Top-5 : {cur_acc5:.2f}%", flush=True)
            
        if total_questions >= 5000: # On teste sur 5000 pour aller vite
            break

    duration = time.time() - start_time
    print(f"\n=== RESULTAT BRUT (FULL MODEL) ===")
    print(f"Accuracy Top-5 : {(top5_correct/total_questions)*100:.2f}%")
    print(f"Questions      : {total_questions}")
    print(f"Vitesse        : {total_questions/duration:.2f} q/s")
    print(f"==================================\n")

if __name__ == "__main__":
    DB_FULL = r"D:\nemlm_v5_3_full.nemdb"
    CORPUS_PATH = r"c:\Users\nemst\Desktop\LLMonCPU\europarl_fr.txt"
    
    data = load_europarl(CORPUS_PATH)
    test_set = data[-5000:] # M\u00eame set de test
    
    evaluate_full(DB_FULL, test_set, None)
