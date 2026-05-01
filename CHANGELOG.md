# Changelog HDC-LLM 📜

## [V0.1.0] - V1 POC (01 Mai 2026)
### Ajouté
- Implémentation de base du HDC (Hyperdimensional Computing).
- Encodage de tokens par vecteurs aléatoires (10k bits).
- Majority Bundling pour la mémorisation de séquences (N-grams).
- Recherche exacte Hamming vectorisée avec NumPy.
### Résultats
- Accuracy@1 sur Train : 93%.
- Latence : 40ms (scan linéaire).

## [V1.0.0] - V2 Industrialisation (01 Mai 2026)
### Ajouté
- **LSH Index** : Recherche approximative en temps constant (10 tables, K=16).
- **Passive-Aggressive Learning** : Apprentissage itératif par correction d'erreur (+2/-1).
- **Subwords** : Encodage par n-grammes de caractères pour la similarité morphologique.
- **Incremental Weighted Sum** : Calcul du bundle en O(D) au lieu de O(N*D).
- **Positional Primes** : Rotations basées sur des nombres premiers pour les contextes longs.
### Améliorations
- Latence réduite de **40ms à 14ms** (D=30k).
- Capacité de mémorisation étendue à des contextes plus larges (N=5).

## [V1.1.0] - Pivot Q&A Sémantique (En cours)
### Ajouté
- **SemanticIndex** : Nouvel encodeur de phrases (Sentence Encoding) pour la similarité.
- **Hybrid Search** : Combinaison LSH + Scan exact en cas d'échec de bucket.
- **QA Engine** : Script interactif pour poser des questions sur un document.
### Résultats
- Premier succès sur Q&A document technique (i3-3220, FX-8350).
- Latence de recherche sémantique : **~20ms** (Python).
