import pandas as pd
import streamlit as st
import pickle
import joblib

# Load the model from the file
loaded_model = joblib.load('realmlp_td_classifier_model.pkl')


st.markdown("# Resultado do modelo RealMLP")
st.markdown("### Abaixo estão os resultados do modelo RealMLP para os pacientes cadastrados. O modelo classifica o risco de doença cardíaca com base nos dados fornecidos.")

def load_model_from_file(file_path):
    import pickle
    with open(file_path, 'rb') as f:
        model = pickle.load(f)
    return model

# Fazer a previsão usando o modelo carregado
# O .predict_proba() retorna as probabilidades para ambas as classes [0, 1]
# Para um classificador binário, geralmente queremos a probabilidade da classe positiva (índice 1)

prediction_probabilities = loaded_model.predict_proba(new_raw_data)[:, 1]

print(f"Probabilidade de Doença Cardíaca: {prediction_probabilities[0]:.4f}")

# Você pode definir um limite para classificar (por exemplo, 0.5)
predicted_class = 1 if prediction_probabilities[0] >= 0.5 else 0
print(f"Classe Predita: {predicted_class}")