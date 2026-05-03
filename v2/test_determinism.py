import numpy as np
from hdc.representation import encode_context, hamming

def test():
    dim = 10000
    ctx = ["monsieur", "le", "président"]
    
    print(f"--- TEST DETERMINISME ENCODE_CONTEXT ---")
    hv1 = encode_context(ctx, dim)
    hv2 = encode_context(ctx, dim)
    
    dist = hamming(hv1, hv2)
    mean_val = np.mean(np.unpackbits(hv1))
    
    print(f"Distance de Hamming entre hv1 et hv2 : {dist}")
    print(f"Densité moyenne du vecteur : {mean_val:.4f}")
    
    if dist == 0:
        print("✅ SUCCESS: encode_context est déterministe.")
    else:
        print("❌ FAILURE: encode_context produit des résultats différents !")

    # Test de stabilité (cache)
    print(f"\n--- TEST STABILITE CACHE (1000 appels) ---")
    dists = []
    for _ in range(1000):
        hv_n = encode_context(ctx, dim)
        dists.append(hamming(hv1, hv_n))
    
    if sum(dists) == 0:
        print("✅ SUCCESS: Le cache est stable.")
    else:
        print(f"❌ FAILURE: Instabilité détectée ({sum(dists)} erreurs sur 1000)")

if __name__ == "__main__":
    test()
