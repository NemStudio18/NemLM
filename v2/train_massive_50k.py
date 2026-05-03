import sys
import os
import time
from hdc.parallel_engine import V3ParallelEngine

CORPUS_PATH = r"c:\Users\nemst\Desktop\LLMonCPU\europarl_fr.txt"
DB_PATH = r"D:\nemlm_v5_3_massive_50k.nemdb"

def load_europarl(path, limit=50000):
    print(f"[*] Chargement de {limit} phrases...")
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= limit: break
            s = line.strip()
            if s:
                data.append(s.split())
    return data

if __name__ == "__main__":
    if os.path.exists(DB_PATH):
        print(f"[*] Suppression de l'ancienne base : {DB_PATH}")
        os.remove(DB_PATH)
        
    sentences = load_europarl(CORPUS_PATH, limit=50000)
    
    # On utilise num_workers=3 (Bigrams, Trigrams, Syntax)
    engine = V3ParallelEngine(dim=10000, db_path=DB_PATH, num_workers=3, use_pruning=True)
    
    print(f"[*] Démarrage de l'entraînement parallèle sur {len(sentences)} phrases...")
    start_time = time.time()
    
    for i, sent in enumerate(sentences):
        engine.train_step(sent)
        if (i + 1) % 5000 == 0:
            print(f"  > Progrès : {i+1}/{len(sentences)} phrases...")
            
    engine.stop()
    end_time = time.time()
    
    print(f"\n[OK] Entraînement massif terminé en {(end_time - start_time)/60:.1f} min")
    print(f"[*] Base générée : {DB_PATH}")
