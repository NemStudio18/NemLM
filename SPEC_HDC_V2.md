# Spécifications HDC-LLM V2.0 🧠

## 1. Vision
Transformer l'approche "Probabilités sur Flottants" (GPU) en "Transitions sur États Binaires" (CPU). L'objectif est d'atteindre une performance de niveau LLM sur du matériel ancien via des opérations XOR, popcount et des structures de données optimisées (BTree, LSH).

## 2. Architecture Technique

### 2.1 Représentation (HDC Core)
- **Dimension (D)** : 100 000 bits par vecteur (par défaut 30 000 pour les tests).
- **Encodage Token** : Combinaison d'un vecteur aléatoire (identité) et d'un bundling de n-grammes de caractères (subwords).
- **Positionnement** : Rotations circulaires basées sur une liste de nombres premiers pour éviter les collisions cycliques dans les séquences longues.

### 2.2 Mémoire Associative
- **Storage** : Dictionnaire de clés HDC (64-bit hashes) vers des entrées de mémoire.
- **Majority Bundling** : Utilisation d'un vecteur de sommes pondérées (Weighted Sum) pour un vote majoritaire bit par bit, préservant l'information fréquentielle.
- **Apprentissage** : Algorithme **Passive-Aggressive (PA)** binaire. Ajustement des poids (+2/-1) si la marge de confiance entre le bon token et le prédit est insuffisante.

### 2.3 Indexation (LSH)
- **Multi-table LSH** : 10 tables de hachage.
- **K-bits** : 16 bits de hachage par table pour une discrimination fine.
- **Complexité** : Recherche en temps constant amorti $O(1)$, indépendant de la taille du vocabulaire.

## 3. Objectif Q&A Sémantique
Le système évolue d'un simple prédicteur vers un moteur de recherche sémantique :
1. **Sentence Encoding** : Fusion des HV des tokens d'une phrase en un seul HV sémantique.
2. **Hamming Similarity** : Utilisation de la distance de Hamming pour trouver les passages d'un document les plus proches d'une question utilisateur.

## 4. Cibles Hardware
- **Léger** : Intel i3-3220 (Ivy Bridge, 2012), 16GB RAM, DDR3.
- **Moyen** : AMD FX-8350, 28GB RAM, DDR3.
- **Contrainte** : Indépendance totale vis-à-vis du GPU.

---
*Version 2.0 - 01 Mai 2026*
