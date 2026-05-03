# Spécification Technique NemLM V5.3 (POC Haute Fidélité)

## Objectif
Démontrer qu'un système Hyperdimensional Computing (HDC) peut rivaliser avec un petit Transformer sur la prédiction du mot suivant, en restant 100% Bitwise sur CPU, sans aucune opération flottante, et avec une inférence < 1.3ms.

## 1. Contraintes Fondamentales
- ✅ **Zéro flottant** en inférence.
- ✅ **Zéro GPU** (Entraînement et Inférence).
- ✅ **Zéro backpropagation** (Phase 1 & 2).
- ✅ **Stockage Industriel** : SQLite (D:) pour gérer des bases de 50 Go+.
- ✅ **Inférence Compacte** : < 1 Go après distillation.

## 2. Architecture HDC-AR (Autorégressif)
La version 5.3 fusionne deux signaux contextuels :

### 2.1 Contexte Local (Syntaxe)
- Basé sur des n-grammes d'ordres 5, 4, 3, et 2.
- Encodage par rotations circulaires et XOR.
- Match exact via SQLite avec **Early Exit** pour privilégier les contextes longs.

### 2.2 Contexte Global (Thématique)
- Géré par le `ContextAccumulator`.
- Somme cumulative des hypervecteurs avec **decay exponentiel (0.95)**.
- Capture la thématique globale de la séquence.

### 2.3 Attention Darwinienne (Fallback)
- 8 têtes d'attention binaire indépendantes.
- Utilise la fusion `Query = Local XOR Global`.
- Sélection par distance de Hamming et survie par utilité (Darwinisme).

## 3. Pipeline de Distillation
Le savoir est compressé de manière pondérée :
- **Top-5 Accuracy** : Seules les 5 meilleures prédictions sont conservées.
- **Relocation** : Les têtes d'attention sont migrées dans la base compacte.
- **Optimisation** : SQLite en mode `query_only` et `mmap` pour la vitesse.

## 4. Métriques Cibles (V5.3)
| Métrique | Cible | État Actuel (Distillé) |
| :--- | :--- | :--- |
| **Accuracy Top-5** | 31.36% | 25.40% |
| **Taux d'Inconnu** | 0.00% | 0.00% |
| **Vitesse Inférence** | > 500 tok/s | ~800 tok/s |
| **Latence** | < 5ms | 1.25ms |
| **RAM Inférence** | < 1 Go | ~500 Mo |

## 5. Philosophie
Toutes les opérations restent bitwise (XOR, rotations, popcount). NemLM démontre qu'il est possible de créer un modèle de langage souverain et local, capable de tourner sur des CPU legacy, tout en maintenant une structure mathématique élégante et frugale.

---
*NemStudio - Advanced Agentic Coding Project*