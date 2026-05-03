import sqlite3
import pickle
import os
import time

def distill_high_fidelity(src_db, dst_db, top_k=30):
    print(f"[*] Distillation ULTRA Fid\u00e9lit\u00e9 (Top-{top_k}) : {src_db} -> {dst_db}")
    
    if os.path.exists(dst_db):
        os.remove(dst_db)
        
    conn_src = sqlite3.connect(src_db)
    conn_dst = sqlite3.connect(dst_db)
    
    cursor_src = conn_src.cursor()
    cursor_dst = conn_dst.cursor()
    
    # Structure de la base compacte
    cursor_dst.execute("CREATE TABLE distilled (context_hv BLOB, token TEXT, weight INTEGER)")
    cursor_dst.execute("CREATE INDEX idx_ctx ON distilled (context_hv)")
    cursor_dst.execute("CREATE TABLE attention (head_id INTEGER, data BLOB)")
    conn_dst.commit()
    
    # 1. Migration de l'Attention (Depuis la table storage)
    print("[*] Extraction des t\u00eates d'attention depuis 'storage'...")
    for i in range(8): # 8 t\u00eates par d\u00e9faut
        key = f"attn_head_{i}".encode()
        cursor_src.execute("SELECT data FROM storage WHERE key = ?", (key,))
        row = cursor_src.fetchone()
        if row:
            cursor_dst.execute("INSERT INTO attention (head_id, data) VALUES (?, ?)", (i, row[0]))
            print(f"  > T\u00eate {i} migr\u00e9e.")
    conn_dst.commit()

    # 2. Distillation des n-grammes (Top-K)
    print(f"[*] Migration des n-grammes (Top-{top_k})...")
    # On exclut les cl\u00e9s d'attention de la migration storage -> distilled
    cursor_src.execute("SELECT key, data FROM storage WHERE key NOT LIKE 'attn_head_%'")
    
    count = 0
    start_time = time.time()
    
    log_file = r"c:\Users\nemst\Desktop\LLMonCPU\v2\result_tests\distill_hf.txt"
    with open(log_file, "w", encoding="utf-8") as log:
        log.write(f"[*] D\u00e9marrage Distillation ULTRA Fid\u00e9lit\u00e9 : {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.flush()
        
        while True:
            rows = cursor_src.fetchmany(1000)
            if not rows: break
            
            for ctx_hv, entry_blob in rows:
                try:
                    entry = pickle.loads(entry_blob)
                    sorted_tokens = sorted(entry.token_weights.items(), key=lambda x: x[1], reverse=True)[:top_k]
                    for token, weight in sorted_tokens:
                        cursor_dst.execute("INSERT INTO distilled (context_hv, token, weight) VALUES (?, ?, ?)", 
                                         (ctx_hv, token, weight))
                except: continue
                
                count += 1
                if count % 10000 == 0:
                    elapsed = time.time() - start_time
                    msg = f"[*] Processed {count} contexts... ({count/elapsed:.1f} ctx/s)"
                    print(msg, flush=True)
                    log.write(msg + "\n")
                    log.flush()
                    conn_dst.commit()
    
    conn_dst.commit()
    print(f"[OK] Distillation termin\u00e9e. {count} contextes trait\u00e9s.")
    
    # Optimisation finale
    print("[*] Optimisation SQLite (VACUUM)...")
    conn_dst.execute("VACUUM")
    conn_dst.close()
    conn_src.close()

if __name__ == "__main__":
    SRC = r"D:\nemlm_v5_3_full.nemdb"
    DST = r"D:\nemlm_v5_3_compact_hf.nemdb"
    
    distill_high_fidelity(SRC, DST, top_k=30)
