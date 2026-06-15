# -- coding: utf-8 --
"""
Script principal d'entrainement du modele BERT
Developpeur B - Devoir BERT Classification
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AdamW
import pandas as pd
from sklearn.metrics import f1_score

# Importation de nos propres modules modules
from utils import fixer_seed, separer_donnees
from dataset import ArxivDataset
from model import ArxivBERTClassifier

def eval_model(model, data_loader, criterion, device):
    """Fonction de validation du modele sur 20% des donnees"""
    model.eval()
    val_loss = 0
    all_targets = []
    all_preds = []
    
    # Obligatoire selon la consigne : pas de calcul de gradient en eval
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            # Transformation des logits brute en predictions binaires (0 ou 1)
            # On applique un seuil a 0.5 apres la fonction Sigmoide
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).int()
            
            all_targets.append(labels.cpu())
            all_preds.append(preds.cpu())
            
    # Concatenation pour calculer les metriques globales
    all_targets = torch.cat(all_targets, dim=0).numpy()
    all_preds = torch.cat(all_preds, dim=0).numpy()
    
    # Calcul du F1-Score macro pour le multi-label
    macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    
    return val_loss / len(data_loader), macro_f1

def main():
    # 1. Configuration initiale
    fixer_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Calcul execute sur : {device}")
    
    # Hyperparametres classiques d'etudiant
    EPOCHS = 3
    BATCH_SIZE = 16
    MAX_LENGTH = 256
    LR = 2e-5 # Petit learning rate recommandé pour BERT
    
    # 2. Preparation des donnees
    df = pd.read_csv("data/train 5.csv")
    df_train, df_val = separer_donnees(df, train_size=0.8, seed=42)
    
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    
    train_dataset = ArxivDataset(df_train, tokenizer, max_length=MAX_LENGTH)
    val_dataset = ArxivDataset(df_val, tokenizer, max_length=MAX_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 3. Initialisation du modele et de l'optimiseur
    model = ArxivBERTClassifier(model_name="google-bert/bert-base-uncased", num_classes=6)
    model.to(device)
    
    # Perte adaptee au multi-label et optimiseur specifique AdamW
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=LR)
    
    best_val_loss = float('inf')
    
    # 4. Boucle d'entrainement principale
    print("\n--- Debut de l'entrainement ---")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if step % 100 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS} | Etape {step}/{len(train_loader)} | Loss: {loss.item():.4f}")
                
        # Evaluation a la fin de chaque epoque
        epoch_train_loss = train_loss / len(train_loader)
        epoch_val_loss, epoch_val_f1 = eval_model(model, val_loader, criterion, device)
        
        print(f"\n>> FIN EPOCH {epoch+1} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val F1 Macro: {epoch_val_f1:.4f}")
        
        # Sauvegarde stricte du meilleur checkpoint selon la consigne
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), "best_bert_model.pt")
            print("[SAUVEGARDE] Meilleur modele enregistre sous 'best_bert_model.pt'\n")

if _name_ == "_main_":
    main()