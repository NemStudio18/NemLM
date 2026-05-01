# 🧠 NemLM (HDC-LLM)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CPU: Legacy Native](https://img.shields.io/badge/CPU-Legacy%20Native-blue.svg)](#)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](#)

**NemLM** est un moteur de langage expérimental basé sur le **Hyperdimensional Computing (HDC)**. Conçu spécifiquement pour les architectures CPU legacy (comme l'Intel i3-3220), NemLM remplace l'attention flottante traditionnelle par des opérations binaires massives (`XOR`, `Popcount`, `Rotations`).

## 🚀 Vision
L'objectif de NemLM est de prouver qu'un modèle de langage capable de généralisation sémantique peut fonctionner sans GPU, en utilisant uniquement les capacités bitwise du processeur.

### Points Forts :
- **Indépendance GPU** : Zéro calcul flottant (FP32/FP16) lors de l'inférence sémantique.
- **Vitesse Bitwise** : Utilisation intensive du produit scalaire binaire et des distances de Hamming.
- **Fusion XOR(C,L)** : Architecture unique combinant Contexte Global (Sémantique) et Contexte Local (N-grams).
- **Empreinte Mémoire** : Stockage compressé (1 bit par poids) via le format binaire `.hdb`.

## 🛠️ Architecture V3 (Génération)
Le moteur NemLM V3 repose sur une fusion autorégressive :
1. **Semantic Layer** : Encodage de phrases par bundling HDC.
2. **Contextual Layer** : Encodage de séquences par rotations circulaires basées sur des nombres premiers.
3. **Associative Memory** : Retrieval ultra-rapide en $O(D)$ via produit matriciel optimisé.

## 📈 Roadmap
- [x] **V1** : Preuve de concept N-grams.
- [x] **V2** : Indexation LSH et Apprentissage Passive-Aggressive.
- [x] **V3** : Prototype de génération conditionnée (Python).
- [ ] **Phase Rust** : Portage natif pour des performances SSE4.2/AVX.
- [ ] **NemLM CLI** : Interface industrielle pour l'entraînement et le test.

## 💻 Installation
```bash
git clone https://github.com/NemStudio18/NemLM.git
cd NemLM
pip install numpy
```

## 🧪 Benchmark
Pour lancer le benchmark de génération massif (300k tokens) :
```bash
python v2/parallel_train.py
```

---
*Développé avec passion pour l'informatique bas niveau et les architectures CPU optimisées.*
