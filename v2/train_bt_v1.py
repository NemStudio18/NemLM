import numpy as np
import time
from hdc.binary_layers import BinaryMLP

# Configuration Ultra-Light
DIM = 512
HIDDEN_DIM = 1024
LR = 0.001
EPOCHS = 10
CORPUS_PATH = r"c:\Users\nemst\Desktop\LLMonCPU\europarl_fr.txt"

def load_data(path, limit=200):
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip().lower().split() for line in f if line.strip()]
    return lines[:limit]

def get_fast_hv(word, dim):
    """Génère un HV binaire déterministe pour le prototype"""
    np.random.seed(hash(word) % (2**32))
    hv = np.random.choice([-1.0, 1.0], size=dim).astype(np.float32)
    return hv

if __name__ == "__main__":
    print(f"[*] Prototype BT Ultra-Light ({DIM}-dim)...")
    model = BinaryMLP(DIM, HIDDEN_DIM)
    
    sentences = load_data(CORPUS_PATH, limit=200)
    output_file = r"c:\Users\nemst\Desktop\LLMonCPU\v2\result_tests\bt_prototype_log.txt"
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"--- BT ULTRA-LIGHT : {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")

    start_time = time.time()
    for epoch in range(EPOCHS):
        total_loss = 0
        steps = 0
        
        for j, sent in enumerate(sentences):
            # Encodage contextuel ultra-simple : Somme des HVs (Sign)
            for i in range(1, len(sent)):
                context_words = sent[max(0, i-3):i]
                target_word = sent[i]
                
                # Input = Sign(Sum(HVs))
                x = np.sign(sum([get_fast_hv(w, DIM) for w in context_words]))
                if isinstance(x, float) or x.shape == (): x = get_fast_hv(sent[i-1], DIM) # Fallback
                
                y_true = get_fast_hv(target_word, DIM)
                
                loss = model.train_step(x, y_true, lr=LR)
                total_loss += loss
                steps += 1
            
            if (j + 1) % 50 == 0:
                msg = f"  > Epoch {epoch+1} | Phrase {j+1}/{len(sentences)} | Loss: {total_loss/steps:.6f}"
                print(msg, flush=True)
                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(msg + "\n")
                
        final_msg = f"[*] Epoch {epoch+1}/{EPOCHS} FINIE | Loss: {total_loss/steps:.6f}"
        print(final_msg, flush=True)
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(final_msg + "\n")

    print(f"\n[OK] Prototype terminé en {time.time() - start_time:.1f}s")
