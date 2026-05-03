import numpy as np
from hdc.v3_engine import V3Engine
from hdc.representation import encode_context

def sanity_check():
    dim = 10000
    engine = V3Engine(dim=dim)
    
    # Phrase de test ultra-simple
    sentence = "le chat mange la souris noire dans la cuisine"
    tokens = sentence.split()
    
    print(f"--- SANITY CHECK PREDICTION ---")
    print(f"Apprentissage de : '{sentence}'")
    engine.train_step(sentence)
    
    # On reconstruit l'index (obligatoire pour predict_topk)
    vocab = list(set(tokens))
    engine.rebuild_lsh(vocab)
    
    # Test 1 : "le chat mange la" -> "souris" ?
    context = ["le", "chat", "mange", "la"]
    target  = "souris"
    
    l_hv = encode_context(context, dim)
    preds = engine.memory.predict_topk(l_hv, k=5)
    
    print(f"Contexte : {context}")
    print(f"Attendu  : {target}")
    print(f"Prédit   : {preds}")
    
    if target in preds:
        print("✅ SUCCESS: NemLM a mémorisé et retrouvé la suite.")
    else:
        print("❌ FAILURE: NemLM est incapable de retrouver une suite simple !")

if __name__ == "__main__":
    sanity_check()
