# Changelog HDC-LLM 📜

## [V0.1.0] - V1 POC (01 Mai 2026)
### Ajouté
- Implémentation de base du HDC (Hyperdimensional Computing).
- Encodage de tokens par vecteurs aléatoires (10k bits).
- Majority Bundling pour la mémorisation de séquences (N-grams).

## [V1.0.0] - V2 Industrialisation (01 Mai 2026)
### Ajouté
- **LSH Index** : Recherche approximative en temps constant (10 tables, K=16).
- **Passive-Aggressive Learning** : Apprentissage itératif par correction d'erreur (+2/-1).
- **Subwords** : Encodage par n-grammes de caractères pour la similarité morphologique.
- **Incremental Weighted Sum** : Calcul du bundle en O(D) au lieu de O(N*D).
- **Positional Primes** : Rotations basées sur des nombres premiers pour les contextes longs.

## [V3.0.0] - Industrialisation Haute Densité (02 Mai 2026)
### Ajouté
- **Bit-Packed Pipeline** : Intégration de `np.packbits` à tous les niveaux (Cache, LSH, Mémoire). Gain de densité : x8.
- **Quantization int8** : Passage de `int16` à `int8` pour les sommes pondérées, divisant par 2 l'empreinte mémoire des contextes.
- **Cache de Rotation (LRU)** : Accélération majeure de l'entraînement par mise en cache des rotations bit-à-bit.
- **RAM Shield (Capacité Bornée)** : Système de pruning automatique à 1M d'entrées pour garantir la stabilité sur 16GB.
- **Spécialisation Française** : Support complet des accents et entraînement sur corpus Europarl (50k+ phrases).
- **Benchmark Scientifique** : Duel NemLM vs Kneser-Ney (5-grammes) pour validation de la précision sémantique.
