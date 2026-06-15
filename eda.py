# -- coding: utf-8 --
"""
Script d'Analyse Exploratoire des Données (EDA)
Developpeur B - Devoir BERT Classification
"""

import pandas as pd
import matplotlib.pyplot as plt

# 1. Chargement du jeu de données
# Le fichier doit etre place dans le dossier data/
path_dataset = "data/train 5.csv"
df = pd.read_csv(path_dataset)

print("==================================================")
print("       ANALYSE EXPLORATOIRE DES DONNEES          ")
print("==================================================")

# 2. Statistiques de base
print(f"Nombre total d'articles dans le dataset : {len(df)}")

# Liste des 6 categories cibles du dataset Arxiv
classes = ['Computer Science', 'Physics', 'Mathematics', 'Statistics', 'Quantitative Biology', 'Quantitative Finance']

# 3. Verification de l'equilibre des classes
print("\n--- Distribution des classes cibles ---")
total_instances = 0
for c in classes:
    nb_articles = df[c].sum()
    pourcentage = (nb_articles / len(df)) * 100
    print(f"• {c} : {nb_articles} articles ({pourcentage:.2f}%)")

# 4. Analyse des longueurs de texte (Titre + Resume)
# On fusionne le titre et le resume comme ce qui sera donne a BERT
df['texte_complet'] = df['TITLE'] + " " + df['ABSTRACT']
df['longueur_mots'] = df['texte_complet'].apply(lambda x: len(str(x).split()))

print("\n--- Statistiques sur la longueur des textes (en mots) ---")
print(f"• Longueur minimale : {df['longueur_mots'].min()} mots")
print(f"• Longueur maximale : {df['longueur_mots'].max()} mots")
print(f"• Longueur moyenne  : {df['longueur_mots'].mean():.2f} mots")
print(f"• Mediane           : {df['longueur_mots'].median()} mots")

# 5. Affichage de quelques exemples (Exigence de la consigne : min 5 exemples ou apercu clair)
print("\n--- Aperçu des 5 premieres lignes du dataset ---")
print(df[['TITLE', 'Computer Science', 'Physics', 'Mathematics']].head(5))

# 6. Generation et sauvegarde du graphique de repartition
plt.figure(figsize=(10, 5))
df[classes].sum().plot(kind='bar', color='skyblue', edgecolor='black')
plt.title("Nombre d'articles par categorie scientifique")
plt.ylabel("Nombre d'articles")
plt.xlabel("Categories")
plt.xticks(rotation=30, ha='right')
plt.tight_layout()

# Sauvegarde de la figure pour le rapport README
plt.savefig("distribution_classes.png")
print("\n[INFO] Graphique sauvegarde sous le nom 'distribution_classes.png'")
print("==================================================")