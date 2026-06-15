# -*- coding: utf-8 -*-
"""
Interface web locale interactive pour tester le modele avec Gradio
Developpeur A - Devoir BERT Classification
"""

import torch
import gradio as gr
from transformers import AutoTokenizer

# Importation de notre modele personnalise
from model import ArxivBERTClassifier

# 1. Configuration de l'environnement et du modele
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classes = ['Computer Science', 'Physics', 'Mathematics', 'Statistics', 'Quantitative Biology', 'Quantitative Finance']

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
model = ArxivBERTClassifier(model_name="google-bert/bert-base-uncased", num_classes=6)

# Chargement des poids entraines (on force le CPU si CUDA n'est pas dispo)
model.load_state_dict(torch.load("best_bert_model.pt", map_location=device))
model.to(device)
model.eval()

def classifier_texte(texte):
    """Fonction qui prend le texte de l'interface, l'encode et retourne les prédictions"""
    if not texte.strip():
        return {c: 0.0 for c in classes}
        
    # Encodage identique a notre classe Dataset
    inputs = tokenizer(
        texte,
        max_length=256,
        padding='max_length',
        truncation=True,
        return_tensors="pt"
    )
    
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        # Utilisation de la fonction Sigmoide pour obtenir une probabilite independante par classe
        probabilites = torch.sigmoid(logits).squeeze(0).cpu().numpy()
        
    # On renvoie un dictionnaire associant chaque categorie a sa probabilite pour Gradio
    return {classes[i]: float(probabilites[i]) for i in range(len(classes))}

# 2. Construction de l'interface web (Structure simple et claire)
demo = gr.Interface(
    fn=classifier_texte,
    inputs=gr.Textbox(lines=6, label="Résumé de l'article scientifique (en anglais) :", placeholder="Saisissez le abstract ici..."),
    outputs=gr.Label(num_top_classes=3, label="Catégories détectées et probabilités :"),
    title="Analyseur d'Articles Scientifiques ArXiv - BERT Multi-Label",
    description="Entrez le titre et le résumé d'une recherche pour identifier ses thématiques majeures.",
    examples=[
        ["Deep Convolutional Neural Networks for Image Classification. In this paper, we present a new architecture to train deep convolutional neural networks on massive vision datasets."],
        ["Stochastic Differential Equations and Financial Markets. This article explores the mathematical modeling of option pricing using advanced probability and variance analysis."]
    ]
)

if __name__ == "__main__":
    # Lancement de l'application locale
    demo.launch()