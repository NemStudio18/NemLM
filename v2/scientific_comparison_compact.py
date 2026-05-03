import sys
import io
import time
import sqlite3
import pickle
import numpy as np
from hdc.corpus import load
from hdc.representation import DIM, encode_context
from eval_kneser_ney import KneserNeyModel

# Fix encodage
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

class CompactHDC:
    def __init__(self, db_path='v2/memory_compact.nemdb'):
        self.conn = sqlite3.connect(db_path)
        
    def predict_topk(self, context_tokens, k=5):
        hv_packed = encode_context(context_tokens, DIM)
        key = hv_packed.tobytes()
        cursor = self.conn.execute("SELECT data FROM storage WHERE key = ?", (key,))
        row = cursor.fetchone()
        if row:
            _, weights = pickle.loads(row[0])
            sorted_tokens = sorted(weights.items(), key=lambda x: x[1], reverse=True)
            return [t[0] for t in sorted_tokens[:k]]
        return []

def run_benchmark():
    print("Chargement des données Europarl...")
    # On utilise le fichier europarl_fr.txt à la racine
    sentences = load("europarl_fr.txt", max_tokens=2000000)
    split = int(len(sentences) * 0.8)
    train_sents = sentences[:split]
    test_sents  = sentences[split:split+5000]

    # 1. Kneser-Ney (Baseline)
    print("Entrainement Kneser-Ney...")
    start = time.time()
    kn_model = KneserNeyModel(n=5)
    kn_model.train(train_sents)
    kn_time = time.time() - start

    # 2. NemLM (HDC Compact)
    print("Chargement NemLM V3 Compact...")
    hdc_model = CompactHDC()

    # 3. ÉVALUATION
    test_size = len(test_sents)
    print(f"Évaluation sur {test_size} questions...")
    hdc_hits = 0
    kn_hits  = 0
    
    for i, sent in enumerate(test_sents):
        if len(sent) < 5: continue
        context = sent[:4]
        target  = sent[4]
        
        hdc_preds = hdc_model.predict_topk(context, k=5)
        kn_preds  = kn_model.predict_topk(context, k=5)
        
        if target in hdc_preds: hdc_hits += 1
        if target in kn_preds: kn_hits += 1
        
        if (i+1) % 500 == 0:
            print(f"Progress: {i+1}/{test_size}")

    print("\n" + "="*40)
    print("RÉSULTATS FINAUX (Top-5 Accuracy)")
    print("="*40)
    print(f"NemLM (HDC) : {hdc_hits/test_size*100:>6.2f}%")
    print(f"Kneser-Ney  : {kn_hits/test_size*100:>6.2f}%")
    print("="*40)

if __name__ == "__main__":
    run_benchmark()
