"""
Test HDC-LLM V3 - Génération Conditionnée
Démontre comment le contexte global (C) guide la génération locale (L).
"""

from hdc.v3_engine import V3Engine

def main():
    print("=== HDC-LLM V3 Test (Conditional Generation) ===")
    engine = V3Engine(dim=30000)

    sentences = [
        "le chat mange une souris dans le jardin",
        "le chien mange un os devant sa niche",
        "le petit oiseau chante sur la branche",
        "le gros poisson nage dans la riviere"
    ]
    
    # Construction du vocabulaire pour le test
    vocab = set()
    for s in sentences:
        vocab.update(s.split())
    vocab = list(vocab)
    
    print(f"[1/2] Entraînement sur {len(sentences)} phrases...")
    for epoch in range(10): # 10 époques pour garantir la convergence
        for s in sentences:
            engine.train_step(s, vocab=vocab)
            
    # 2. Test de génération conditionnée
    prompts = ["le chat", "le chien", "le petit", "le gros"]
    
    print("\n[2/2] Résultats de génération :")
    for p in prompts:
        engine.long_term_hvs = [] 
        gen = engine.generate(p, max_new_tokens=6, vocab=vocab)
        print(f"  Prompt: '{p}'")
        print(f"  Sortie: '{p} {' '.join(gen)}'")
        print("-" * 30)

if __name__ == "__main__":
    main()
