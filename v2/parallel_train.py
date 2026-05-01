"""
HDC-LLM Parallel Trainer - Massive Scale (6000 Phrases)
Cible : 300 000 tokens, 50% CPU Load.
"""

import multiprocessing as mp
import time
import os
import sys

if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from hdc.v3_engine import V3Engine
from hdc.corpus import load, build_vocabulary
from hdc.memory import AssociativeMemory

def train_chunk(chunk_id, sentences_chunk, dim, vocab):
    print(f"Worker {chunk_id} : Demarrage sur {len(sentences_chunk)} phrases...")
    engine = V3Engine(dim=dim)
    for i, s_tokens in enumerate(sentences_chunk):
        # On utilise l'entrainement standard
        # predict_topk est maintenant accelere par np.dot
        engine.train_step(" ".join(s_tokens), vocab=vocab)
        if (i+1) % 500 == 0:
            print(f"Worker {chunk_id} : {i+1}/{len(sentences_chunk)} phrases traitees.")
    return engine.memory

def main():
    print("\n" + "="*50)
    print(" HDC-LLM MASSIVE SCALE TEST (300k TOKENS) ".center(50, "="))
    print("="*50 + "\n")

    num_workers = 2 
    sentences_raw = load("../v1/text8")
    # 6000 phrases = ~300 000 tokens
    sentences_raw = sentences_raw[:6000] 
    vocab = build_vocabulary(sentences_raw)
    dim = 30000

    print(f"Stats Corpus : {len(sentences_raw)} phrases, {len(vocab)} mots uniques.")
    print(f"Configuration : {num_workers} workers (50% CPU).")

    chunk_size = len(sentences_raw) // num_workers
    chunks = [sentences_raw[i : i + chunk_size] for i in range(0, len(sentences_raw), chunk_size)]

    print(f"Lancement de l'entrainement massif...")
    t0 = time.perf_counter()
    
    with mp.Pool(processes=num_workers) as pool:
        results = [pool.apply_async(train_chunk, (i, chunks[i], dim, vocab)) for i in range(num_workers)]
        memories = [res.get() for res in results]
            
    t1 = time.perf_counter()
    total_time = t1 - t0
    
    print(f"\n✅ Termine en {total_time:.2f} s")
    print(f"🚀 Debit Global : {len(sentences_raw)*50/total_time:.1f} tokens/sec")

    print("🧠 Fusion des memoires...")
    master_memory = AssociativeMemory(dim=dim)
    for m in memories:
        master_memory.merge(m)
    
    master_memory.save("model_massive.hdb")
    print(f"💾 Taille finale : {master_memory.size} entrees.")
    print(f"📦 Fichier : {os.path.getsize('model_massive.hdb')/1024/1024:.2f} MB")

if __name__ == "__main__":
    main()
