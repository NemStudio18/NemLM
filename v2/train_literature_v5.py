import time
import os
import numpy as np
from hdc.v3_engine import V3Engine
from hdc.representation import DIM
from hdc.corpus import tokenize

def format_time(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}m {s:02d}s"

def log_print(msg, log_file):
    print(msg, flush=True)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

def load_literary_corpus(files, log_file):
    all_sentences = []
    log_print(f"[*] Chargement du corpus litteraire...", log_file)
    for f_path in files:
        if os.path.exists(f_path):
            with open(f_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Tokenization simple par ligne/phrase
                sentences = [s.strip().split() for s in content.split('.') if len(s.strip().split()) > 3]
                all_sentences.extend(sentences)
                log_print(f"  > {os.path.basename(f_path)} : {len(sentences)} phrases", log_file)
    return all_sentences

def run_training():
    log_file = "literature_bench.txt"
    # Reset log file
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("=== NemLM V5.2 Literature Training Log ===\n")

    log_print("="*60, log_file)
    log_print("      NEMLM V5.2 - ENTRAINEMENT LITTERAIRE (HDC-AR)", log_file)
    log_print("="*60, log_file)
    
    corpus_files = [
        "../corpus_fr/monte_cristo.txt",
        "../corpus_fr/les_miserables.txt",
        "../corpus_fr/bel_ami.txt"
    ]
    
    sentences = load_literary_corpus(corpus_files, log_file)
    np.random.shuffle(sentences) # Melange pour un apprentissage homogene
    
    # Split Train/Eval
    eval_size = 1000
    train_sents = sentences[:-eval_size]
    eval_sents = sentences[-eval_size:]
    
    db_path = r"D:\nemlm_literature_v5.nemdb"
    engine = V3Engine(DIM, db_path=db_path)
    
    log_print(f"\n[*] Base de donnees : {db_path}", log_file)
    log_print(f"[*] Phrases d'entrainement : {len(train_sents)}", log_file)
    log_print(f"[*] Phrases d'evaluation   : {len(eval_sents)}", log_file)
    log_print("-" * 60, log_file)

    chunk_size = 2000
    start_all = time.time()
    
    for i in range(0, len(train_sents), chunk_size):
        chunk = train_sents[i : i + chunk_size]
        log_print(f"\n[>>>] Debut du Bloc {i//chunk_size + 1} ({len(chunk)} phrases)...", log_file)
        start_chunk = time.time()
        
        # 1. Entrainement du bloc avec logs granulaires
        for j, sent in enumerate(chunk):
            engine.train_step(sent)
            if (j + 1) % 500 == 0:
                log_print(f"  > Traite {i + j + 1}/{len(train_sents)} sentences...", log_file)
        
        log_print(f"[*] Fin du bloc. Commit en cours...", log_file)
        engine.commit()
        
        # 2. Evaluation rapide (Benchmark)
        log_print(f"[*] Benchmark en cours sur {min(len(eval_sents), 100)} phrases...", log_file)
        hits = 0
        total_eval = 0
        for sent in eval_sents[:100]: # Reduit a 100 pour la vitesse
            if len(sent) < 5: continue
            context = sent[:4]
            target = sent[4]
            
            preds = engine.predict_next(context, top_k=5)
            if target in preds:
                hits += 1
            total_eval += 1
        
        accuracy = (hits / total_eval * 100) if total_eval > 0 else 0
        
        # 3. Logs et ETA
        elapsed_all = time.time() - start_all
        speed = (i + len(chunk)) / (elapsed_all + 0.1)
        eta = (len(train_sents) - (i + len(chunk))) / speed
        
        log_print(f"Bloc {i//chunk_size + 1:3d} | "
                  f"Progress: {min(100, (i+chunk_size)/len(train_sents)*100):5.1f}% | "
                  f"Acc@5: {accuracy:5.2f}% | "
                  f"Speed: {speed:5.1f} phr/s | "
                  f"ETA: {format_time(eta)}", log_file)

    log_print("\n" + "="*60, log_file)
    log_print("             ENTRAINEMENT TERMINE", log_file)
    log_print("="*60, log_file)
    log_print(f" Temps total : {format_time(time.time() - start_all)}", log_file)
    log_print(f" Accuracy finale : {accuracy:.2f}%", log_file)
    log_print("="*60, log_file)

if __name__ == "__main__":
    run_training()
