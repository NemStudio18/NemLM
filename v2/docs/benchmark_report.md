# Rapport de Benchmark Scientifique - NemLM V5

## Configuration du Duel V5 (En cours)
- **Modèle** : NemLM V3 (Hybrid Attention)
- **Baseline** : Kneser-Ney 5-gram (Statistical)
- **Corpus** : Europarl French (25 000 phrases, ~1.2M tokens)
- **Architecture** : HDC 10 000 bits, 8 Attention Heads, 1024 Semantic Keys.
- **Stockage** : SQLite sur SSD (D:), Cache RAM 2 Go.

## État de l'Apprentissage (Snapshot 23:26)
- **Progression** : 14 500 / 25 000 phrases (58%)
- **Temps par tranche (500)** : ~63 secondes
- **Vitesse moyenne** : 8 phrases / sec
- **Stabilité RAM** : Excellente (~1.8 Go constants)
- **Taille BDD estimée** : ~35-40 Go (en fin de run)

## Objectifs Scientifiques
1. Valider que l'**Attention Binaire** permet une meilleure généralisation que KN sur des contextes jamais vus.
2. Démontrer la scalabilité du moteur HDC sur CPU avec un stockage disque haute performance.
3. Quantifier l'impact du repli sémantique (HDC) par rapport au repli statistique (KN).

---
*Dernière mise à jour : 2026-05-02 23:26*
