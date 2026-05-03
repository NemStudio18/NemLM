# Spécifications NemLM V5.3+ (HDC-AR)

## 1. Encodage & Représentation
- **Dimension** : 10 000 bits (Bit-packed en `uint8`).
- **Encodage Positionnel** : Rotations circulaires bitwise.
- **ContextAccumulator** : Somme pondérée avec decay (0.95) et binarisation par signe.

## 2. Moteur d'Inférence (Phase 2)
- **Multi-scale Backoff** : Vote pondéré (ordres 5, 4, 3, 2).
- **Pondération syntaxique** : `Score = count * n**4`.
- **Early Exit** : Sortie précoce si certitude statistique (n=5, count > 2).
- **Distillation** : Compression Top-30 (Fidélité vs Taille).

## 3. Raisonnement Différentiable (Phase 3B)
- **Binary MLP** : Neurones binaires (-1, 1).
- **Apprentissage STE** : Straight-Through Estimator pour le gradient binaire.
- **Gradient Clipping** : Stabilisation globale L2 (η = 1.0).
- **Entrée Hybride** : `Input = Context_HV XOR Attention_HV`.

---
*NemStudio - Advanced Agentic Coding Project*
