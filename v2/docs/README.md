# NemLM : LLM Frugal & 100% Bitwise (V5.3 Multi-Worker)

NemLM est une architecture expérimentale de modèle de langage (LLM) conçue pour s'affranchir totalement des GPU et des calculs flottants. Basé sur le **Calcul Hyperdimensionnel (HDC)**, NemLM utilise des opérations binaires (XOR, Popcount, Rotations) pour simuler des capacités de raisonnement et de génération de texte.

## 🚀 Vision & Objectifs
- **Performance GPT-2 Small** : Atteindre la qualité d'un modèle de 117M paramètres sur un CPU legacy.
- **Frugalité Extrême** : Fonctionne avec **4-6 Go de RAM** et sur n'importe quel processeur (x64/ARM).
- **Zéro Flottant** : L'inférence est 100% binaire (Bitwise).
- **Vitesse** : Cible d'inférence de **< 1ms/token**.

## 🧠 Architecture HDC-AR (V5.3)
La version actuelle utilise une architecture **Autorégressive Hyperdimensionnelle (HDC-AR)** parallélisée :
- **Multi-Worker Training** : Entraînement asynchrone sur 3 cœurs CPU (3 specialized workers).
- **Singleton Pruning** : Filtrage des associations uniques pour maximiser la densité sémantique.
- **Multi-scale Backoff** : Prédiction précise via n-grammes d'ordres 5, 4, 3, et 2.
- **Thematic Accumulator** : Un accumulateur global pour la cohérence sémantique long-terme.

## 🛠️ Stack Technique
- **Core** : Python 3.10+ (Optimisé Numpy / SQLite).
- **Persistence** : SQLite (Mode WAL + mmap) pour gérer des bases de données de plus de 100 Go sur SSD.
- **Primitives** : XOR, Bit-Packing, Popcount Table Lookup.

## 📊 Résultats du Duel (V5.3 Milestone)
- **Accuracy (Top-5)** : **31.36%** (NemLM) vs **32.80%** (Kneser-Ney).
- **Vitesse Entraînement** : **230+ phr/s** (Parallel Multi-Worker).
- **Taille Base de Données** : **~450 Mo** (filtrée) vs ~1.8 Go (non-filtrée).

### 🛠️ Conditions du Duel
Pour garantir une validité scientifique, le test a été réalisé dans les conditions suivantes :
- **Corpus** : Europarl FR (Débats du Parlement Européen).
- **Dataset** : 15 000 phrases pour l'entraînement, 5 000 phrases pour le test.
- **Hardware** : Intel i3-3220 @ 3.30GHz (Architecture de 2012).
- **RAM** : Consommation stable à **~1.2 Go** (incluant le cache SQLite).
- **Contrainte** : NemLM utilise des vecteurs de **10 000 bits** et des opérations 100% binaires. Kneser-Ney utilise des calculs flottants de probabilités classiques (interpolation).

---
*Projet conçu et développé par NemStudio dans le cadre de l'Advanced Agentic Coding.*
