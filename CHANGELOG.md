# Changelog - NemLM (LLMonCPU)

## [V5.4-dev] - 2026-05-03 (Binary Transformer Milestone)
### Ajouté
- **BT Prototype (Phase 3B)** : Implémentation réussie du premier neurone différentiable binaire (STE).
- **Convergence STE** : Validation de la descente de gradient sur des poids binarisés (Sign).
- **Gradient Clipping** : Stabilisation de la rétropropagation pour le calcul bitwise.

## [V5.3] - 2026-05-03 (High-Fidelity Industrial Milestone)
### Ajouté
- **Validation Industrielle HF** : Accuracy Top-5 atteinte de **32.03%** sur Europarl (15k), atteignant la parité avec Kneser-Ney.
- **Calibration n^4** : Optimisation de la pondération multi-ordre pour une meilleure séparation syntaxique.
- **CompactEngine (HDC-AR)** : Moteur d'inférence ultra-léger (~1.38 Go) atteignant ~110 tokens/s avec une latence < 10ms.
- **Distillation Haute Fidélité** : Processus de compression Top-30 préservant la longue traîne statistique.

## [V5.2] - 2026-05-03 (HDC-AR Milestone)
### Ajouté
- **HDC-AR (Autorégressif)** : Nouvelle architecture fusionnant contexte Local (syntaxe) et Global (thématique).
- **ContextAccumulator** : Gestion de la cohérence long-terme avec decay exponentiel (0.95).
- **Multi-scale Backoff** : Recherche pondérée sur n-grammes d'ordres 5, 4, 3, et 2.

---
*NemStudio - Advanced Agentic Coding Project*
