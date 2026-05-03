import time
import os
from hdc.parallel_engine import V3ParallelEngine

def load_europarl(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        sentences = f.readlines()
    return [s.strip().lower() for s in sentences]

def train_full(corpus_path, db_path, output_file, dim=10000, train_size=15000):
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"[*] Base existante supprim\u00e9e : {db_path}")

    with open(output_file, "w", encoding="utf-8") as log:
        msg = f"[*] D\u00e9marrage Entra\u00eenement Full (No Pruning) - {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        log.write(msg); log.flush(); print(msg, end="")
        
        msg = f"[*] Chargement du corpus ({train_size} phrases)...\n"
        log.write(msg); log.flush(); print(msg, end="")
        data = load_europarl(corpus_path)[:train_size]
        
        msg = f"[*] Initialisation du V3ParallelEngine...\n"
        log.write(msg); log.flush(); print(msg, end="")
        engine = V3ParallelEngine(dim, db_path, use_pruning=False)
        
        start_time = time.time()
        
        for i, sentence in enumerate(data):
            tokens = sentence.split()
            engine.train_step(tokens)
            
            if (i + 1) % 1000 == 0:
                elapsed = time.time() - start_time
                speed = (i + 1) / elapsed
                msg = f"  > Progress: {((i+1)/train_size)*100:5.1f}% | Speed: {speed:6.1f} phr/s\n"
                log.write(msg); log.flush(); print(msg, end="")
                
        msg = "[*] Finalisation de l'entrainement (Attente des workers)...\n"
        log.write(msg); log.flush(); print(msg, end="")
        engine.stop()
        
        duration = time.time() - start_time
        final_msg = f"\n[OK] Entra\u00eenement termin\u00e9 en {duration:.1f}s.\n"
        final_msg += f"[*] Taille finale de la base : {os.path.getsize(db_path)/(1024*1024):.2f} Mo\n"
        log.write(final_msg); log.flush(); print(final_msg)

if __name__ == "__main__":
    CORPUS_PATH = r"c:\Users\nemst\Desktop\LLMonCPU\europarl_fr.txt"
    DB_FULL = r"D:\nemlm_v5_3_full.nemdb"
    OUTPUT_LOG = r"c:\Users\nemst\Desktop\LLMonCPU\v2\result_tests\train_full_v5_3.txt"
    
    if not os.path.exists(os.path.dirname(OUTPUT_LOG)):
        os.makedirs(os.path.dirname(OUTPUT_LOG))
        
    train_full(CORPUS_PATH, DB_FULL, OUTPUT_LOG)
