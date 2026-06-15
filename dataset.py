# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset

class ArxivDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=256):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Liste des colonnes cibles
        self.label_cols = ['Computer Science', 'Physics', 'Mathematics', 
                           'Statistics', 'Quantitative Biology', 'Quantitative Finance']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # On combine le titre et le résumé pour donner plus de contexte à BERT
        text = str(row['TITLE']) + " " + str(row['ABSTRACT'])
        
        # Tokenisation avec les paramètres classiques
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Extraction des labels sous forme de FloatTensor (obligatoire pour le multi-label)
        labels = torch.tensor(row[self.label_cols].values, dtype=torch.float)
        
        # On extrait les tenseurs (on squeeze pour enlever la dimension de batch parasite de size 1)
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': labels
        }