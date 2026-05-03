import time
import os
import numpy as np
from hdc.v3_engine import V3Engine
from hdc.representation import encode_context

def load_europarl(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        sentences = f.readlines()
    return [s.strip().lower() for s in sentences]

def train_attention_only(corpus_path, db_path, output_file, train_size=15000):
    with open(output_file, "w", encoding="utf-8") as log:
        msg = f"[*] D\u00e9marrage Entra\u00eenement Attention : {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        log.write(msg); log.flush(); print(msg, end="")
        
        msg = f"[*] Chargement du moteur V3 (DB: {db_path})\n"
        log.write(msg); log.flush(); print(msg, end="")
        engine = V3Engine(db_path=db_path)
        
        msg = f"[*] Chargement du corpus ({train_size} phrases)...\n"
        log.write(msg); log.flush(); print(msg, end="")
        data = load_europarl(corpus_path)[:train_size]
        
        start_time = time.time()
        
        msg = "[*] Entra\u00eenement de l'Attention Head (Darwinian Learning)...\n"
        log.write(msg); log.flush(); print(msg, end="")
        
        for i, sentence in enumerate(data):
            tokens = sentence.split()
            if len(tokens) < 2: continue
            
            engine.accumulator.reset()
            for j in range(1, len(tokens)):
                context = tokens[max(0, j - 5):j]
                target = tokens[j]
                
                l_hv = encode_context(context, engine.dim)
                g_hv = engine.accumulator.get_hv()
                query_hv = np.bitwise_xor(l_hv, g_hv)
                
                target_hv = engine.semantic.get_word_hv(target)
                engine.attention.learn(query_hv, target_hv)
                engine.accumulator.add(target_hv)
                
            if (i + 1) % 1000 == 0:
                elapsed = time.time() - start_time
                msg = f"  > Progr\u00e8s: {((i+1)/train_size)*100:5.1f}% | Vitesse: {(i+1)/elapsed:6.1f} phr/s\n"
                log.write(msg); log.flush(); print(msg, end="")
                
        msg = "[*] Sauvegarde de l'Attention dans la DB...\n"
        log.write(msg); log.flush(); print(msg, end="")
        engine.commit()
        
        duration = time.time() - start_time
        final_msg = f"[OK] Attention entra\u00een\u00e9e et sauvegard\u00e9e en {duration:.1f}s.\n"
        log.write(final_msg); log.flush(); print(final_msg)

if __name__ == "__main__":
    CORPUS_PATH = r"c:\Users\nemst\Desktop\LLMonCPU\europarl_fr.txt"
    DB_FULL = r"D:\nemlm_v5_3_full.nemdb"
    OUTPUT_LOG = r"c:\Users\nemst\Desktop\LLMonCPU\v2\result_tests\train_attention_v5_3.txt"
    
    if not os.path.exists(os.path.dirname(OUTPUT_LOG)):
        os.makedirs(os.path.dirname(OUTPUT_LOG))
        
    train_attention_only(CORPUS_PATH, DB_FULL, OUTPUT_LOG)
