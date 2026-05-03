import sys
import nltk
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import KneserNeyInterpolated

CORPUS_PATH = r"c:\Users\nemst\Desktop\LLMonCPU\europarl_fr.txt"

def load_europarl(path, limit=55000):
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip().lower().split() for line in f if line.strip()]
    return lines[:limit]

if __name__ == "__main__":
    data = load_europarl(CORPUS_PATH)
    train_data = data[:50000]
    test_data = data[50001:55000]
    
    print(f"[*] Entraînement KN sur 50 000 phrases...")
    train, vocab = padded_everygram_pipeline(5, train_data)
    kn_model = KneserNeyInterpolated(5)
    kn_model.fit(train, vocab)
    
    print(f"[*] Evaluation KN sur 5 000 phrases...")
    total = 0
    correct_top5 = 0
    
    for tokens in test_data:
        if len(tokens) < 5: continue
        for i in range(2, len(tokens)):
            context = tuple(tokens[max(0, i-4):i])
            target = tokens[i]
            
            # Get top 5
            preds = [p[0] for p in kn_model.context_counts(kn_model.vocab.lookup(context)).most_common(5)]
            if target in preds:
                correct_top5 += 1
            total += 1
            
            if total % 1000 == 0:
                print(f"  > Progrès : {total} tokens | Accuracy Top-5 : {correct_top5/total*100:.2f}%")
            if total >= 10000: break
        if total >= 10000: break
        
    print(f"\n=== RAPPORT KNESER-NEY (50k) ===")
    print(f"Accuracy Top-5 : {correct_top5/total*100:.2f}%")
