# Addendum n° 1 à la fiche de gel : phase B2a (carte des régimes de la KD)

**Rédigé : ____ / ____ / 2026**   **Daté et validé (avant tout lancement) : ____ / ____ / 2026**   **Signature : ____________**

Cet addendum complète `fiche_gel_protocole_v1.md`. Il ne modifie aucune règle de la Phase B. Rien de ce qui suit n'a été calculé au moment de la rédaction : aucune seed de B2a n'a été lancée.

## 1. Statut de la phase B2a

- B2a a été **planifiée après avoir vu les résultats de la Phase B** (registre des écarts, écarts n° 2 et n° 3). Elle sert à rendre **confirmatoire** la carte des régimes de la KD, qui était exploratoire avec 5 seeds par condition.
- Elle utilise des **seeds jamais utilisées** (15 à 24), les **mêmes teachers sauvegardés** et le **même code d'entraînement** que la Phase B.
- Elle se rapporte **en plus** de la Phase B, jamais à sa place. Les résultats de la Phase B restent dans l'article tels quels.

## 2. Question et hypothèses

**Question (RQ1).** Pour chacune des 27 conditions (3 datasets × 3 teachers × 3 students), la KD de référence (Vanilla : T = 4, α = 0,5) améliore-t-elle le Macro-F1 de test du student, est-elle sans effet, ou le dégrade-t-elle ?

**Hypothèses**, écrites avant tout calcul. Seuil δ = 0,02 de Macro-F1 (le même que dans la fiche) ; les classes A, B, C sont définies en section 5.

- **HB1.** Sur FractureMNIST3D, la KD améliore le F1 dans les 9 conditions. *Soutenue* si les 9 conditions sont de classe A et si le gain moyen des 9 conditions dépasse δ ; *partiellement soutenue* si 6 à 8 conditions sont de classe A ; *réfutée* sinon.
- **HB2.** Sur OrganMNIST3D, la KD n'a pas d'effet établi. *Soutenue* si aucune condition n'est de classe A ou C et si au moins 7 conditions sur 9 sont de classe B ; *réfutée* si au moins une condition est de classe A ou C ; *partiellement soutenue* sinon.
- **HB3** (exploratoire, la plus risquée). Sur AdrenalMNIST3D, un gain n'existe qu'avec le plus petit student S1. *Soutenue* si les 3 conditions avec S1 sont de classe A et si aucune des 6 conditions avec S2 ou S3 n'est de classe A ; *réfutée* sinon. Le couple pivot (res_w32_b2 × S1, 15 seeds) suggère déjà un gain proche de 0 avec S1 : une réfutation est probable et sera rapportée.

## 3. Plan d'expériences

| Bloc | Contenu | Runs |
| --- | --- | ---: |
| Student seul | 3 students × 3 datasets × seeds 15 à 24 | 90 |
| Vanilla (KD de référence) | 9 couples teacher–student × 3 datasets × seeds 15 à 24 | 270 |
| **Total** | | **360** |

- Même configuration que la Phase B : 80 epochs, AdamW (lr 1e-3, weight decay 1e-4), batch 32, schedule cosinus, augmentation sans flips, normalisation avec les statistiques du train, checkpoint choisi sur la NLL de validation, 28³.
- **Teachers inchangés** : les fichiers `teacher_<nom>.pt` de la Phase B, gelés. Leur empreinte SHA-256 est enregistrée au lancement (commande : `sha256sum outputs/phaseB_*/teacher_*.pt`).
- **Protection du test** : évalué une seule fois par run, sur le meilleur checkpoint de validation, jamais affiché dans les logs, jamais utilisé pour choisir quoi que ce soit.
- Temps estimé : environ 2 h sur 2 GPU (estimation à partir des durées mesurées : 10 à 14 s par Student seul ; 26 s, 46 à 56 s et 74 s par run de KD avec res_w16_b1, res_w32_b1 et res_w32_b2).
- Le plan est implémenté par un nouveau plan `b2a` du script `09_phase_b.py`, avec les mêmes gardes (doublons, nombre d'epochs, reprise). Il est testé avec `--smoke` avant le lancement.

## 4. Seeds et appariement

Seeds 15 à 24, soit 10 seeds par condition, identiques pour le Student seul et pour Vanilla (appariement par seed : même initialisation, même ordre des données, même augmentation). **Aucune seed supplémentaire ne sera ajoutée après avoir vu les résultats de B2a.** Si un run est interrompu par un problème d'infrastructure, il est relancé avec la même seed et le même code, et l'incident est noté.

## 5. Analyse et règles de classement

Pour chaque condition (dataset, teacher, student) :

- **Gain** = Macro-F1 de test de Vanilla − Macro-F1 de test du Student seul, par seed ; moyenne, écart-type, intervalle de confiance à 95 % de Student (t, 9 degrés de liberté), p-valeur du test t apparié.
- **Correction de Holm** par dataset, sur les 9 conditions du dataset (famille de taille 9).
- **Classes**, dans cet ordre :
  - **A, la KD aide** : l'intervalle est entièrement au-dessus de 0 et le p ajusté (Holm) est inférieur à 0,05 ;
  - **C, la KD nuit** : l'intervalle est entièrement en dessous de 0 et le p ajusté est inférieur à 0,05 ;
  - **B, sans effet** : l'intervalle est entièrement dans [−0,02 ; +0,02] ;
  - **non classé** : tous les autres cas.
- Un drapeau signale les conditions de classe A dont l'intervalle est entièrement dans ]0 ; 0,02] (gain établi mais négligeable).
- Intervalle bootstrap (percentile, 10 000 réechantillonnages, graine fixe) rapporté en complément.
- **Métriques secondaires** (balanced accuracy, AUC, ECE, NLL de test) : gain et intervalle, descriptifs, sans verdict ni correction.
- **Robustesse, descriptive** : pour chaque condition, la classe obtenue est comparée au gain moyen de F1 des 7 autres schedules de la Phase B (5 seeds). Aucun verdict n'est tiré de cette comparaison.
- **Pas de test global** agrégeant les 27 conditions : les décomptes par classe sont descriptifs ; les neuf conditions d'un dataset partagent trois teachers et trois students.
- Variables de régime (descriptives, jamais utilisées pour définir les classes) : F1 du Student seul sur la validation, qualité du teacher, ratio de capacité.

## 6. Ce qui est interdit après la signature

- Ajouter des seeds, des conditions ou des méthodes à B2a après avoir vu les résultats.
- Modifier δ, la famille de Holm, les règles de classement ou les hypothèses.
- Utiliser le test pour choisir un teacher, un student, un checkpoint, ou pour modifier le plan.
- Relancer un run uniquement parce que son résultat déplaît (seules les interruptions d'infrastructure justifient une relance).
- Retirer des résultats de l'article parce qu'ils réfutent une hypothèse.

## 7. Traçabilité à enregistrer au lancement

- Empreintes SHA-256 des 9 teachers, de `09_phase_b.py` (avec le plan `b2a`) et du script d'analyse de B2a.
- Version des bibliothèques (PyTorch, NumPy, SciPy), matériel (2 × RTX A4000), date et heure du lancement.
- Écart éventuel avec cet addendum (à noter dans le registre des écarts de la fiche, section 13).

## 8. À vérifier avant de dater

- [ ] Les hypothèses HB1 à HB3 et leurs critères sont ceux que tu veux, sans y changer un mot après avoir regardé de nouveaux résultats.
- [ ] δ = 0,02, Holm par dataset sur 9 conditions, intervalle de Student : conformes à la fiche de gel.
- [ ] Le plan `b2a` du script est écrit et testé avec `--smoke`.
- [ ] Le script d'analyse de B2a est écrit et testé sur de faux CSV.
- [ ] Le registre des écarts de la fiche (section 13) mentionne l'écart n° 2 (recadrage après les résultats), qui motive B2a.
