"""
HDC-LLM Parallel Trainer 
Permet de choisir l'echelle et le nombre de coeurs.
"""

import multiprocessing as mp
import time
import os
import sys
import argparse

if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from hdc.v3_engine import V3Engine
from hdc.corpus import load, build_vocabulary
from hdc.memory import AssociativeMemory

def train_chunk(chunk_id, sentences_chunk, dim, vocab):
    """Fonction executee par chaque processus."""
    print(f"Worker {chunk_id} : Demarrage sur {len(sentences_chunk)} phrases...")
    engine = V3Engine(dim=dim)
    for i, s_tokens in enumerate(sentences_chunk):
        engine.train_step(" ".join(s_tokens), vocab=vocab)
        if (i+1) % 500 == 0:
            print(f"Worker {chunk_id} : {i+1}/{len(sentences_chunk)} phrases traitees.")
    return engine.memory

def main():
    parser = argparse.ArgumentParser(description="Entraîneur HDC-LLM NemLM")
    parser.add_argument("--phrases", type=int, default=2000, help="Nombre de phrases a traiter")
    parser.add_argument("--workers", type=int, default=2, help="Nombre de processus paralleles")
    parser.add_argument("--dim", type=int, default=30000, help="Dimension des hypervecteurs")
    parser.add_argument("--output", type=str, default="model_nemlm.hdb", help="Nom du fichier de sortie")
    args = parser.parse_args()

    print("\n" + "="*50)
    print(" NemLM PARALLEL TRAINER ".center(50, "="))
    print("="*50 + "\n")

    # 1. Configuration
    num_workers = args.workers
    print(f"Configuration : {num_workers} workers.")
    
    sentences_raw = load("../v1/text8")
    sentences_raw = sentences_raw[:args.phrases] 
    vocab = build_vocabulary(sentences_raw)
    dim = args.dim

    print(f"Corpus : {len(sentences_raw)} phrases, {len(vocab)} mots.")

    # 2. Split
    chunk_size = len(sentences_raw) // num_workers
    chunks = [sentences_raw[i : i + chunk_size] for i in range(0, len(sentences_raw), chunk_size)]

    # 3. Training Parallele
    print(f"Lancement de l'entrainement...")
    t0 = time.perf_counter()
    
    with mp.Pool(processes=num_workers) as pool:
        results = [pool.apply_async(train_chunk, (i, chunks[i], dim, vocab)) for i in range(num_workers)]
        memories = [res.get() for res in results]
            
    t1 = time.perf_counter()
    total_time = t1 - t0
    print(f"Entrainement termine en {total_time:.2f} s")
    print(f"Debit : {len(sentences_raw)*50/total_time:.1f} tokens/sec")

    # 4. Fusion des Cerveaux
    print("Fusion des memoires...")
    master_memory = AssociativeMemory(dim=dim)
    for m in memories:
        master_memory.merge(m)
    
    master_memory.save(args.output)
    print(f"Modele sauvegarde : {args.output} ({os.path.getsize(args.output)/1024:.1f} KB)")

if __name__ == "__main__":
    main()
