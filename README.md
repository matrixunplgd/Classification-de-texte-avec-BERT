# 📚 Classification Multi-Label d'Articles Scientifiques ArXiv avec BERT

## 👥 Membres du Binôme
* **Développeur A :** alamine (`matrixunplgd`) - Implémentation du Dataset PyTorch, Architecture du Modèle & Interface Démo (Gradio)
* **Développeur B :** Damstat / movicsarr - Analyse Exploratoire des Données (EDA), Reproductibilité (Seed) & Boucle d'Entraînement principale

---

## 📑 Présentation du Projet
Ce projet consiste à concevoir un système de classification automatique d'articles scientifiques issus de la base **arXiv** parmi 6 disciplines majeures :
1. Computer Science
2. Physics
3. Mathematics
4. Statistics
5. Quantitative Biology
6. Quantitative Finance

Puisqu'un article scientifique peut être transverse (par exemple, traiter à la fois de Mathématiques et de Finance Quantitative), nous avons mis en place une architecture de **classification multi-label** (prédiction non exclusive). Le modèle choisi est le transformeur pré-entraîné **BERT** (`bert-base-uncased`), finetuné en PyTorch.

---

## 📊 1. Analyse Exploratoire des Données (`eda.py`)
Le script d'analyse exploratoire extrait et sauvegarde les statistiques indispensables du jeu de données `train 5.csv` :
* **Volume total :** Comptage du nombre total de lignes et d'instances de données.
* **Équilibre des classes :** Calcul de la distribution sémantique et du pourcentage exact de représentation de chaque catégorie cible pour mettre en évidence le déséquilibre du dataset.
* **Analyse de texte :** Fusion textuelle (Titre + Abstract) et calcul de la longueur moyenne des documents (en mots).
* **Visualisation :** Génération et sauvegarde automatique du graphique `distribution_classes.png` pour illustrer visuellement la répartition.

---

## ⚙️ 2. Prétraitement & Reproductibilité (`dataset.py` & `utils.py`)
* **Tokenisation :** Utilisation de `AutoTokenizer` (Hugging Face) pour transformer le texte brut en tokens compatibles avec BERT, avec une gestion stricte du padding et de la troncature à une dimension de `max_length=256`.
* **Graine Aléatoire (Reproductibilité) :** Fixation systématique de la seed (`seed=42`) sur toutes les couches (`random`, `numpy`, `torch.manual_seed` pour CPU et CUDA) pour assurer que le professeur obtienne exactement les mêmes métriques lors de sa correction.
* **Data Split :** Séparation robuste et déterministe du dataset en **80% Train** pour l'apprentissage et **20% Validation** pour le contrôle des performances.

---

## 🧠 3. Architecture du Modèle (`model.py`)
L'architecture utilise l'encodeur BERT comme extracteur de caractéristiques (Backbone).
* Récupération du vecteur d'état caché du token spécial **`[CLS]`** situé à la première position (`outputs.last_hidden_state[:, 0, :]`), qui encapsule la représentation sémantique globale de l'abstract.
* Ajout d'une tête de classification linéaire (`nn.Linear(768, 6)`) projetant l'espace de BERT vers nos 6 classes cibles.
* **Fonction de Perte :** Utilisation de `nn.BCEWithLogitsLoss()`. Contrairement au multi-classe (Softmax), cette perte applique une Sigmoïde indépendante sur chaque neurone de sortie, ce qui est obligatoire pour le traitement multi-label.

---

## ⚡ 4. Entraînement & Évaluation (`train.py`)
Le script gère la mécanique d'apprentissage en pur PyTorch :
* **Optimiseur :** `AdamW` avec un taux d'apprentissage adapté aux transformeurs (`2e-5`).
* **Métriques de performance :** Suivi de la Loss et calcul du **F1-Score Macro** à chaque époque (la métrique la plus fiable pour évaluer un modèle multi-label asymétrique).
* **Mécanisme de Checkpoint :** Sauvegarde stricte du meilleur modèle sous le format `best_bert_model.pt` basé sur la baisse de la loss de validation.

---

## 🌐 5. Interface Graphique Interactive (`demo.py`)
Déploiement d'une application de démonstration locale via la bibliothèque **Gradio**. Elle permet de tester interactivement le modèle entraîné en saisissant le résumé d'un article pour observer en temps réel les probabilités graphiques des catégories associées.