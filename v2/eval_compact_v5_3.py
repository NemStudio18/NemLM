import time
import os
from hdc.compact_engine import CompactEngine

def load_europarl(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        sentences = f.readlines()
    return [s.strip().lower() for s in sentences]

def evaluate_compact(db_path, test_data, output_file, train_limit=15000):
    print(f"[*] Evaluation du moteur compact : {db_path}")
    engine = CompactEngine(db_path)
    
    top1_correct = 0
    top5_correct = 0
    total_questions = 0
    unk_count = 0
    
    start_time = time.time()
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(f"\n\n--- NOUVELLE EVALUATION (HF Top-30) : {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        f.write(f"Param\u00e8tres : n^5 weighting, No Early Exit, Distillation Top-30\n")
    
    for sentence in test_data:
        tokens = sentence.split()
        if len(tokens) < 5: continue
        
        engine.reset_context()
        for i in range(2, len(tokens)):
            context = tokens[:i]
            target = tokens[i]
            
            # On re-synchronise l'accumulateur sur le VRAI pass\u00e9 (Teacher Forcing)
            # Doit correspondre EXACTEMENT a l'entrainement (mots 1 a i-1)
            engine.reset_context()
            for prev_token in tokens[1:i]: 
                engine.accumulator.add(engine.semantic.get_word_hv(prev_token))
            
            preds = engine.predict_next(context, top_k=5)
            
            if not preds or preds[0] == "<unk>":
                unk_count += 1
            else:
                if target == preds[0]:
                    top1_correct += 1
                if target in preds:
                    top5_correct += 1
            
            total_questions += 1
            if total_questions % 500 == 0:
                cur_acc5 = (top5_correct / total_questions) * 100
                msg = f"[*] Progr\u00e8s : {total_questions} questions | Accuracy Top-5 : {cur_acc5:.2f}%"
                print(msg, flush=True)
                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(msg + "\n")
                    f.flush()
            
        if total_questions >= 10000:
            break

    duration = time.time() - start_time
    
    acc1 = (top1_correct / total_questions) * 100
    acc5 = (top5_correct / total_questions) * 100
    unk_rate = (unk_count / total_questions) * 100
    
    report = f"""=== RAPPORT D'EVALUATION NEMLM COMPACT ===
Date : {time.strftime("%Y-%m-%d %H:%M:%S")}
Base : {db_path}

--- METRIQUES ---
Questions test\u00e9es : {total_questions}
Accuracy Top-1   : {acc1:.2f}%
Accuracy Top-5   : {acc5:.2f}%
Taux d'Inconnu   : {unk_rate:.2f}% (Mod\u00e8le muet)

--- PERFORMANCE ---
Temps total      : {duration:.1f}s
Vitesse moyenne  : {total_questions/duration:.2f} questions/s
ms par question  : {duration*1000/total_questions:.2f} ms
==========================================
"""
    print(report)
    
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(report)
    print(f"[OK] Rapport enregistr\u00e9 dans : {output_file}")

if __name__ == "__main__":
    DB_COMPACT = r"D:\nemlm_v5_3_massive_50k_hf.nemdb"
    CORPUS_PATH = r"c:\Users\nemst\Desktop\LLMonCPU\europarl_fr.txt"
    OUTPUT = r"c:\Users\nemst\Desktop\LLMonCPU\v2\result_tests\compact_eval_v5_3_massive.txt"
    
    # TEST DE PREUVE : Utiliser la base 50k sur le set de test où on faisait 32.55%
    TRAIN_LIMIT = 50000
    TEST_START = 15001
    TEST_COUNT = 2000
    
    if not os.path.exists(os.path.dirname(OUTPUT)):
        os.makedirs(os.path.dirname(OUTPUT))
        
    data = load_europarl(CORPUS_PATH)
    # On prend les 5000 phrases APRES les 50000 d'entraînement
    test_set = data[TEST_START:TEST_START + TEST_COUNT]
    
    evaluate_compact(DB_COMPACT, test_set, OUTPUT, train_limit=TRAIN_LIMIT)
