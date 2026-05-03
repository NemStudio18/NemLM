import sys
from collections import Counter, defaultdict

def train_kneser_ney(sentences, n=5):
    counts = [defaultdict(Counter) for _ in range(n + 1)]
    for tokens in sentences:
        for i in range(1, n + 1):
            for j in range(len(tokens) - i + 1):
                ngram = tuple(tokens[j:j+i])
                prefix = ngram[:-1]
                target = ngram[-1]
                counts[i][prefix][target] += 1
    return counts

def predict_kn(counts, context, n=5, top_k=5):
    for i in range(n, 1, -1):
        prefix = tuple(context[-(i-1):]) if i > 1 else ()
        if prefix in counts[i]:
            preds = counts[i][prefix].most_common(top_k)
            return [p[0] for p in preds]
    return []

CORPUS_PATH = r"c:\Users\nemst\Desktop\LLMonCPU\europarl_fr.txt"

def load_europarl(path, limit=55000):
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip().lower().split() for line in f if line.strip()]
    return lines[:limit]

if __name__ == "__main__":
    data = load_europarl(CORPUS_PATH)
    train_data = data[:50000]
    test_data = data[50001:55000]
    
    print(f"[*] Entraînement KN Natif sur 50 000 phrases...")
    kn_counts = train_kneser_ney(train_data, n=5)
    
    print(f"[*] Evaluation KN sur 5 000 phrases (10 000 tokens)...")
    total = 0
    correct_top5 = 0
    
    for tokens in test_data:
        if len(tokens) < 5: continue
        for i in range(2, len(tokens)):
            context = tokens[:i]
            target = tokens[i]
            
            preds = predict_kn(kn_counts, context, n=5, top_k=5)
            if target in preds:
                correct_top5 += 1
            total += 1
            
            if total % 1000 == 0:
                print(f"  > Progrès : {total} tokens | Accuracy Top-5 : {correct_top5/total*100:.2f}%")
            if total >= 10000: break
        if total >= 10000: break
        
    print(f"\n=== RAPPORT KNESER-NEY NATIVE (50k) ===")
    print(f"Accuracy Top-5 : {correct_top5/total*100:.2f}%")
