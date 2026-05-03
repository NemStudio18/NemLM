import sqlite3
import pickle
import os
import time

# On importe MemoryEntry pour pouvoir d\u00e9s\u00e9rialiser les objets pickle
from hdc.memory import MemoryEntry

def distill(src_db, dest_db, output_file, top_k=5):
    if os.path.exists(dest_db):
        os.remove(dest_db)
    
    src_conn = sqlite3.connect(src_db)
    dest_conn = sqlite3.connect(dest_db)
    dest_conn.execute("PRAGMA journal_mode = WAL")
    dest_conn.execute("CREATE TABLE IF NOT EXISTS distilled (id BLOB PRIMARY KEY, preds BLOB)")
    dest_conn.execute("CREATE TABLE IF NOT EXISTS attention (id BLOB PRIMARY KEY, data BLOB)")
    
    with open(output_file, "w", encoding="utf-8") as log:
        msg = f"[*] D\u00e9marrage Distillation : {src_db} -> {dest_db}\n"
        log.write(msg); log.flush(); print(msg, end="")
        
        # Copie de l'Attention Head
        msg = "[*] Copie des t\u00eates d'Attention...\n"
        log.write(msg); log.flush(); print(msg, end="")
        cursor_attn = src_conn.execute("SELECT key, data FROM storage WHERE key LIKE 'attn_head_%'")
        for k, d in cursor_attn:
            dest_conn.execute("INSERT OR REPLACE INTO attention VALUES (?, ?)", (k, d))
        dest_conn.commit()
        
        msg = "[*] Distillation des n-grammes (Top-K)...\n"
        log.write(msg); log.flush(); print(msg, end="")
        
        cursor = src_conn.execute("SELECT key, data FROM storage WHERE key NOT LIKE 'attn_head_%'")
        
        count = 0
        start_time = time.time()
        
        for hv_key, data in cursor:
            try:
                entry = pickle.loads(data)
                if not entry.token_weights: continue
                    
                sorted_tokens = sorted(entry.token_weights.items(), key=lambda x: x[1], reverse=True)
                top_preds = sorted_tokens[:top_k]
                
                dest_conn.execute("INSERT OR REPLACE INTO distilled VALUES (?, ?)", 
                                 (hv_key, pickle.dumps(top_preds)))
                
                count += 1
                if count % 50000 == 0:
                    elapsed = time.time() - start_time
                    msg = f"  > Distilled: {count} entries | Speed: {count/elapsed:.1f} entries/s\n"
                    log.write(msg); log.flush(); print(msg, end="")
            except:
                continue
                
        dest_conn.commit()
        msg = "[*] Optimisation finale (VACUUM)...\n"
        log.write(msg); log.flush(); print(msg, end="")
        dest_conn.execute("VACUUM")
        
        duration = time.time() - start_time
        final_msg = f"[*] Distillation termin\u00e9e en {duration:.1f}s.\n"
        final_msg += f"[*] Entr\u00e9es distill\u00e9es : {count}\n"
        final_msg += f"[*] Taille finale : {os.path.getsize(dest_db)/(1024*1024):.2f} Mo\n"
        log.write(final_msg); log.flush(); print(final_msg)

    src_conn.close()
    dest_conn.close()

if __name__ == "__main__":
    SRC = r"D:\nemlm_v5_3_full.nemdb"
    DEST = r"D:\nemlm_v5_3_compact_full.nemdb"
    LOG = r"c:\Users\nemst\Desktop\LLMonCPU\v2\result_tests\distill_v5_3_full.txt"
    
    distill(SRC, DEST, LOG)
