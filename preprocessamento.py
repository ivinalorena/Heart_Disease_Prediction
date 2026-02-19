import torch
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from pytabkit import RealMLP_TD_Classifier # Jupyter com informações sobre o realmlp
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder

warnings.filterwarnings('ignore')

# Configuração global
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RANDOM_STATE = 42
N_FOLDS = 5

print(f"Using device: {DEVICE}")

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
original = pd.read_csv("Heart_Disease_Prediction.csv") 

le = LabelEncoder() # classe utilitaria usada para converter texto categórico ou rótulos não numéricos em números inteiros
train['Heart Disease'] = le.fit_transform(train['Heart Disease'])
original['Heart Disease'] = le.fit_transform(original['Heart Disease'])

base_features = [col for col in train.columns if col not in ['Heart Disease', 'id']] 

def add_engineered_features(df):
    df_temp = df.copy()
    
    for col in base_features:  # verifica se existe a feature no df original
        if col in original.columns:
           
            stats = original.groupby(col)['Heart Disease'].agg(['mean', 'median', 'std', 'skew', 'count']).reset_index()
            """ Para cada valor único da coluna col, calcula estatísticas da variável alvo 'Heart Disease':
            Média
            Mediana
            Desvio padrão
            Assimetria (skew)
            Contagem """

            stats.columns = [col] + [f"orig_{col}_{s}" for s in ['mean', 'median', 'std', 'skew', 'count']]
     
            df_temp = df_temp.merge(stats, on=col, how='left') # add as estatísticas calculadas ao DataFrame df_temp através de um left join.
 
            fill_values = { # tratamento de valores ausentes
                f"orig_{col}_mean": original['Heart Disease'].mean(),
                f"orig_{col}_median": original['Heart Disease'].median(),
                f"orig_{col}_std": 0,
                f"orig_{col}_skew": 0,
                f"orig_{col}_count": 0
            }
            df_temp = df_temp.fillna(value=fill_values)
            
    return df_temp

train = add_engineered_features(train)
test = add_engineered_features(test) 

X = train.drop(['id', 'Heart Disease'], axis=1)
y = train['Heart Disease']
X_test = test.drop(['id'], axis=1)

""" 
X: features de treino (remove 'id' e a variável alvo)
y: variável alvo (Heart Disease)
X_test: features de teste (apenas remove 'id') """