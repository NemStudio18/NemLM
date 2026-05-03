import sqlite3
import pickle
import time
import os

SRC_DB = r"D:\nemlm_v5_3_massive_50k.nemdb"
DST_DB = r"D:\nemlm_v5_3_massive_50k_hf.nemdb"

def distill():
    if os.path.exists(DST_DB): os.remove(DST_DB)
    
    conn_src = sqlite3.connect(SRC_DB)
    conn_dst = sqlite3.connect(DST_DB)
    
    conn_dst.execute("CREATE TABLE distilled (context_hv BLOB, token TEXT, weight INTEGER)")
    conn_dst.execute("CREATE TABLE attention (head_id INTEGER PRIMARY KEY, data BLOB)")
    conn_dst.execute("CREATE INDEX idx_ctx ON distilled(context_hv)")
    
    print("[*] Distillation de la mémoire (Top-30 HF)...")
    cursor = conn_src.execute("SELECT key, data FROM storage")
    
    count = 0
    for key, data in cursor:
        entry = pickle.loads(data)
        # On garde les 30 meilleurs
        sorted_tokens = sorted(entry.token_weights.items(), key=lambda x: x[1], reverse=True)[:30]
        
        for token, weight in sorted_tokens:
            conn_dst.execute("INSERT INTO distilled VALUES (?, ?, ?)", (key, token, weight))
        
        count += 1
        if count % 10000 == 0: print(f"  > Contextes traités : {count}")
    
    print("[*] Migration de l'Attention...")
    # S'il y a une table attention dans la source, on la copie (si le script v3_engine l'a créée)
    try:
        cursor_attn = conn_src.execute("SELECT head_id, data FROM attention")
        for row in cursor_attn:
            conn_dst.execute("INSERT INTO attention VALUES (?, ?)", row)
    except:
        print("  ! Pas de table attention trouvée dans la source.")

    conn_dst.commit()
    conn_src.close()
    conn_dst.close()
    print(f"[OK] Distillation terminée : {DST_DB}")

if __name__ == "__main__":
    distill()
