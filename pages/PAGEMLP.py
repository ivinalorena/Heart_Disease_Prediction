import streamlit as st

btn1,btn2 = st.columns(2, gap="medium",border=False, width=400,vertical_alignment='bottom')
with btn1:
    if st.button("Página inicial"):
        st.switch_page("main.py")

with btn2:
    if st.button("Modelo LSTM"):
        st.switch_page("pages/PAGELSTM.py")
st.divider()

st.markdown("# Modelo escolhido: RealMLP")
st.markdown("""### Informações relevantes sobre o modelo escolhido na aba atual: 
Real valued Multi-Layer Perceptron - é um modelo de deep learning avançado, projetado para superar o Gradiente Boosted Decision Trees (XGBoost - CatBoost - LightGBM) em dados tabulares.
- é um conjunto otimizado de técnicas aplicado a um perceptron multicamadas simples.
- é um algoritmo de aprendizado supervisionado que aprende uma função *f:R^m -> R^0* treinando em um conjunto de dados, onde o m é o número de dimensões para entrada e é o O número de dimensões para saída.
- Dado um conjunto de características X = {x1,x2,...,xm} e um alvo y, ele pode aprender um aproximador de função não linear para classificação ou regressão.
- É diferente da regressão logística, pois entre a camada de entrada e a de saída, pode haver uma ou mais camadas não lineares, chamadas camadas ocultas. """) #,height=200, key="system_prompt")

with st.form(key='form_MLP', border=True):
    nome = st.text_input('Nome')
    idade = st.number_input('Idade',format='%d',max_value=100,value=None)
    file = st.file_uploader('Insira os dados do paciente',accept_multiple_files=False,type="csv")
    
    if file is not None: 
        st.dataframe(file)

    if nome or idade or file is None:
        st.warning('Complete o cadastro')
    st.form_submit_button('Cadastrar')

