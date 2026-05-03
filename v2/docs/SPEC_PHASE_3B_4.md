# Spécifications : Phase 3B & Phase 4 (Reasoning NemLM)

Ce document détaille l'architecture prévue pour les étapes de raisonnement complexe et de généralisation.

## 🚀 Phase 3B : Binary Transformer (BT)
L'objectif est d'ajouter une couche différentiable binaire pour permettre la généralisation hors-distribution.

### 3B.1 Architecture Hybride (Validated Prototype)
- **Input Fusion** : Le Binary Transformer reçoit une entrée combinée : `X = Context_HV ⊕ Attention_HV`. 
- **Raisonnement Augmenté** : Cette fusion permet au modèle de comparer la tendance sémantique (contexte) avec les faits précis retrouvés en mémoire (Attention).
- **MLP Binaire** : Couches denses utilisant uniquement des additions et des soustractions via **STE (Straight-Through Estimator)**.
- **Quantification 1-bit** : Les poids et activations sont binarisés par la fonction `sign(x)`.

### 3B.2 Optimisation (Binary Backprop)
- **STE (Straight-Through Estimator)** : Permet de passer le gradient à travers la fonction `sign` non-dérivable pendant l'entraînement.
- **Latent Weights** : On maintient des poids en haute précision (float32) pendant l'apprentissage, mais on utilise les poids binarisés (-1, 1) en inférence.

---

## 🔮 Phase 4 : RAG HDC (Reasoning & Retrieval)
L'objectif est d'implémenter un raisonnement chaîné via des recherches itératives dans la mémoire épisodique.

### 4.1 Orchestrateur : Reasoning Accumulator (Raisonnement Natif)
Le raisonnement n'est pas un module externe, mais un processus itératif utilisant le `ContextAccumulator` existant.

**Algorithme de Raisonnement Chaîné :**
1. **Initialisation** : La question `Q` est injectée dans l'accumulateur.
2. **Boucle de Réflexion** (max 5 steps) :
   - On récupère l'Hypervecteur consolidé : `current_hv = Accumulator.get_hv()`.
   - On interroge la mémoire avec ce contexte enrichi : `Result_n = Memory.retrieve(current_hv)`.
   - On ré-injecte `Result_n` dans l'accumulateur pour orienter la prochaine itération.
3. **Convergence Hamming (Critère d'Arrêt Natif)** :
   - Le processus s'arrête quand `Hamming(HV_n, HV_{n-1}) < 12.5%`. 
   - Cette convergence signifie que NemLM a "fait le tour" du sujet et que les nouvelles informations ne modifient plus sa perception thématique.
4. **Synthèse Finale** : Vote majoritaire sur l'ensemble des résultats accumulés.

> [!TIP]
> Ce mécanisme permet l'émergence d'une "chaîne de pensée" (Chain-of-Thought) sans aucun paramètre flottant ni règle codée en dur.

### 4.2 Mémoire Épisodique SQLite
- **Long-Term Storage** : Stockage des chaînes de pensée (CoT) sous forme de séquences d'hypervecteurs.
- **Similarity Search** : Utilisation de la distance de Hamming pour retrouver des schémas de raisonnement analogues.

### 4.3 Grounding Factuel (Evidence Injection)
- **Mécanisme** : Weighted Bundling (Superposition pondérée).
- **Injection** : Le `fact_hv` n'est pas XORé (ce qui inverserait les bits), mais fusionné dans l'Attention via un vote majoritaire pondéré : `New_Query = Sign(w1*Query + w2*Fact)`. Cela permet au fait de "guider" la recherche sans détruire le signal initial.

---
*NemStudio - Advanced Agentic Coding Project*
