import streamlit as st
import pandas as pd
# https://www.kaggle.com/datasets/cdeotte/s6e4-original-dataset
# 
df = pd.read_csv('C:\\Users\\Ivina\\Desktop\\heartDisease\\data\\Heart_Disease_Prediction.csv').head()
# app princial
#if home_page:
st.header("Predição - Doença do coração")
st.markdown("""
            ### App desenvolvido para:
            - Analisar e prever a presença de doença cardíaca.
            ### Estrutura: 
            - Estrutura tabular: cada linha representa um paciente e cada coluna uma medida médica ou indicador de diagnóstico.
            """)

with st.container(border=True, horizontal_alignment= "center"):
    st.write("Estrutura geral: ")
    st.dataframe(df)
    #st.divider()
    #st.write("Infos de cada coluna: ")
    c1,c2,c3,c4,c5,c6,c7 = st.tabs(["Age","Sex","Chest pain type","BP","Cholesterol","FBS over 120","EKG results"])
    c8,c9,c10,c11,c12,c13,c14 = st.tabs(["Max HR","Exercise angina","ST depression","Slope of ST","Number of vessels fluro","Thallium","Heart Disease"])
    with c1:
        st.write("Idade dos pacientes")
    with c2:
        st.write("Gênero do paciente (1 = Male, 0 = Female)")
    with c3:
        st.write("Tipo de dor no peito: 1 = Angina típica (Typical angina) || 2 = Angina atípica (Atypical angina) || 3 = Dor não anginos(Non-anginal pain) = Assintomática (Asymptomatic) ")
    with c4:
        st.write("Pressão arterial em repouso (mm Hg)")
    with c5:
        st.write("Nível de colesterol sérico (mg/dL)")
    with c6:
        st.write("Glicemia em jejum > 120 mg/dL (1 = Verdadeiro, 0 = Falso) ")
    with c7:
        st.write("Resultados do eletrocardiograma em repouso: 0 = Normal || 1 = Anormalidade da onda ST-T || 2 = Hipertrofia ventricular esquerda")
    with c8:
        st.write("Frequência cardíaca máxima alcançada")
    with c9:
        st.write("Angina induzida por exercício (1 = Sim, 0 = Não)")
    with c10:
        st.write("Depressão do segmento ST induzida pelo exercício em relação ao repouso")
    with c11:
        st.write("Inclinação do segmento ST no pico do exercício")
    with c12:
        st.write("Número de vasos principais (0–3) visualizados por fluoroscopia")
    with c13:
        st.write("Resultado do teste de esforço com tálio (indicador médico categórico)")
    with c14:
        st.write("Variável alvo: Presença = Doença cardíaca detectada - Ausência = Sem doença cardíaca")

st.subheader("Selecione o modelo:")
col1, col2 = st.columns(2) 
with col1:
    realMLP = st.button("⚛ RealMLP", key='RealMLP',type='secondary')
    if realMLP:
        st.switch_page("pages/page_realMLP.py")

with col2:
    model_LSTM = st.button("⚛  LSTM",key='LSTM_btn', type='secondary')
    st.divider()

""" with st.sidebar:
    st.header("✚ Heart Disease Prediction")
    #sidebar = st.sidebar.selectbox("Selecione um modelo treinado para prever",['RealMLP','LSTM'])
    #home_page = st.button("🏠︎ Página Inicial")
    st.subheader("Selecione o modelo:")
    col1, col2 = st.columns(2) 
    with col1:
        realMLP = st.button("⚛ RealMLP", key='RealMLP',type='secondary')
    with col2:
        model_LSTM = st.button("⚛  LSTM",key='LSTM_btn', type='secondary')
    st.divider() """


if model_LSTM:
    st.markdown("# Modelo escolhido: LSTM")
    st.markdown("""### Informações relevantes sobre o modelo escolhido no app atual: **LSTM**
    #### Real valued Multi-Layer Perceptron - é um modelo de deep learning avançado, projetado para superar o Gradiente Boosted Decision Trees (XGBoost - CatBoost - LightGBM) em dados tabulares.
    - é um conjunto otimizado de técnicas aplicado a um perceptron multicamadas simples.
    - é um algoritmo de aprendizado supervisionado que aprende uma função *f:R^m -> R^0* treinando em um conjunto de dados, onde o m é o número de dimensões para entrada e é o O número de dimensões para saída.
    - Dado um conjunto de características X = {x1,x2,...,xm} e um alvo y, ele pode aprender um aproximador de função não linear para classificação ou regressão.
    - É diferente da regressão logística, pois entre a camada de entrada e a de saída, pode haver uma ou mais camadas não lineares, chamadas camadas ocultas. """)

st.divider()
st.caption("Dash desenvolvido por Ivina Lorena")