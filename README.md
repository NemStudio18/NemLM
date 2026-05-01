# 🧠 NemLM (HDC-LLM)

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![CPU: Legacy Native](https://img.shields.io/badge/CPU-Legacy%20Native-orange.svg)](#)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](#)

**NemLM** est un moteur de langage expérimental basé sur le **Hyperdimensional Computing (HDC)**. Conçu spécifiquement pour les architectures CPU legacy (comme l'Intel i3-3220), NemLM remplace l'attention flottante traditionnelle par des opérations binaires massives (`XOR`, `Popcount`, `Rotations`).

## 🚀 Vision
L'objectif de NemLM est de prouver qu'un modèle de langage capable de généralisation sémantique peut fonctionner sans GPU, en utilisant uniquement les capacités bitwise du processeur.

### Points Forts :
- **Indépendance GPU** : Zéro calcul flottant (FP32/FP16) lors de l'inférence sémantique.
- **Vitesse Bitwise** : Utilisation intensive du produit scalaire binaire et des distances de Hamming.
- **Fusion XOR(C,L)** : Architecture unique combinant Contexte Global (Sémantique) et Contexte Local (N-grams).
- **Licence AGPL v3** : Logiciel libre et open-source.

## 🛠️ Architecture V3
Le moteur NemLM V3 repose sur une fusion autorégressive :
1. **Semantic Layer** : Encodage de phrases par bundling HDC.
2. **Contextual Layer** : Encodage de séquences par rotations circulaires.
3. **Associative Memory** : Retrieval ultra-rapide via produit matriciel optimisé.

## 📈 Statistiques de Performance (Python Baseline)
*Benchmarks en cours de calcul sur Intel i3-3220 (2 Cores / 4 Threads)*
- **Débit Entraînement** : En cours d'évaluation...
- **Latence Inférence** : En cours d'évaluation...
- **Capacité** : Test massif de 300 000 tokens en cours.

## 💻 Utilisation
```bash
# Entraîner le modèle avec des paramètres personnalisés
python src/train.py --phrases 1000 --workers 2
```

## 🏗️ Structure du Projet
- `src/` : Moteur NemLM V3 (Cerveau, Mémoire, Représentation).
- `legacy/` : Prototypes V1 et V2 (Recherche sémantique).
- `data/` : Corpus de test (Text8).

---
*Développé sous licence AGPL v3 pour la communauté du calcul frugal.*
