# -- coding: utf-8 --
"""
Fonctions utilitaires : Graine aleatoire et Split des donnees
Developpeur B - Devoir BERT Classification
"""

import torch
import random
import numpy as np

def fixer_seed(seed=42):
    """Fixe toutes les graines aleatoires pour avoir exactement les memes resultats"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Parametres pour forcer PyTorch a rester deterministe
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[INFO] Graine aleatoire fixee a : {seed}")

def separer_donnees(df, train_size=0.8, seed=42):
    """Decoupe de maniere simple et reproductible le dataset en Train / Validation"""
    # On melange les lignes de facon stable grace a random_state
    df_melange = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Calcul du point de bascule
    limite = int(len(df_melange) * train_size)
    
    df_train = df_melange.iloc[:limite].reset_index(drop=True)
    df_val = df_melange.iloc[limite:].reset_index(drop=True)
    
    print(f"[INFO] Dataset divise : {len(df_train)} train, {len(df_val)} validation")
    return df_train, df_val