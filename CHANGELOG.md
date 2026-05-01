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

## [V1.3.0] - Persistance & Stabilité V3 (01 Mai 2026)
### Ajouté
- **Binary Persistence (Option B)** : Format `.hdb` compact [Hash64|BitsPacked] pour une portabilité totale vers Rust.
- **Stable Context Management** : Correction du moteur de génération pour préserver le contexte sémantique (C) entre les prompts.
- **Fast Loader** : Chargement instantané des Hypervecteurs packés depuis le disque.
### Améliorations
- **V3 Engine** : Unification du guidage sémantique et local pour une meilleure cohérence.
- **Memory Optimization** : Utilisation de `np.packbits` pour diviser par 8 la taille des fichiers de modèles.
