# -*- coding: utf-8 -*-
"""
Definition du modele de classification basé sur BERT
Developpeur A - Devoir BERT Classification
"""

import torch
import torch.nn as nn
from transformers import AutoModel

class ArxivBERTClassifier(nn.Module):
    def __init__(self, model_name="google-bert/bert-base-uncased", num_classes=6):
        super(ArxivBERTClassifier, self).__init__()
        
        # On recupere le modele de base BERT (le "backbone")
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Couche de classification finale (768 neurones en entree pour BERT base)
        self.classifier = nn.Linear(768, num_classes)
        
    def forward(self, input_ids, attention_mask):
        # Passage du texte d'entree dans les couches de BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Extraction du premier token spécial [CLS] qui contient le resume sémantique
        # Forme de last_hidden_state : [batch_size, sequence_length, 768]
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        # Calcul des scores brute (logits) pour chaque classe
        # On ne met pas de Softmax/Sigmoid ici, c'est la fonction de perte qui le fera
        logits = self.classifier(cls_output)
        
        return logits