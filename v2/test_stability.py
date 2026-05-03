import numpy as np
from hdc.representation import encode_context, hamming

def test():
    dim = 10000
    # Situation d'entrainement (fenetre de 5)
    ctx_train = ["je", "voudrais", "monsieur", "le", "président"]
    
    # Situation de generation (sentence entiere mais slicee a 5)
    full_sentence = ["aujourd'hui", "je", "voudrais", "monsieur", "le", "président"]
    ctx_gen = full_sentence[-5:]
    
    print(f"Ctx Train: {ctx_train}")
    print(f"Ctx Gen  : {ctx_gen}")
    
    hv_train = encode_context(ctx_train, dim)
    hv_gen   = encode_context(ctx_gen, dim)
    
    dist = hamming(hv_train, hv_gen)
    print(f"Hamming Distance: {dist}")
    
    if dist == 0:
        print("✅ SUCCESS: Les contextes sont stables.")
    else:
        print("❌ FAILURE: Les contextes divergent encore !")

if __name__ == "__main__":
    test()
