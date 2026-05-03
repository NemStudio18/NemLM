import time
from hdc.compact_engine import CompactEngine

def test_inference(db_path, prompts):
    print(f"[*] Initialisation du CompactEngine avec {db_path}...")
    engine = CompactEngine(db_path)
    
    for prompt in prompts:
        print(f"\n[Prompt] : {prompt}")
        tokens = prompt.split()
        
        start_time = time.time()
        generated = []
        
        # Gnration de 15 tokens
        for _ in range(15):
            preds = engine.predict_next(tokens + generated, top_k=1)
            next_token = preds[0]
            if next_token == "<unk>": break
            generated.append(next_token)
            
        if len(generated) > 0:
            duration = time.time() - start_time
            full_text = prompt + " " + " ".join(generated)
            
            print(f"[NemLM]  : {full_text}")
            print(f"[Vitesse]: {len(generated)/duration:.2f} tokens/s ({duration*1000/len(generated):.1f} ms/token)")
        else:
            print(f"[NemLM]  : (Aucune suite trouv\u00e9e dans la base distill\u00e9e)")

if __name__ == "__main__":
    DB_COMPACT = r"D:\nemlm_v5_3_compact.nemdb"
    PROMPTS = [
        "le parlement europ\u00e9en",
        "la commission doit",
        "les droits de l'homme",
        "nous devons agir"
    ]
    test_inference(DB_COMPACT, PROMPTS)
