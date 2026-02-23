import streamlit as st
import time
import pandas as pd
from Controllers import paciente_controller as pac_con

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
    st.subheader("Cadastro de paciente para análise")
    nome = st.text_input('Nome')
    arquivo = st.file_uploader('Insira os dados do paciente', accept_multiple_files=False, type="csv")

    df = None
    if arquivo is not None:
        df = pd.read_csv(arquivo,header=0,index_col=False)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        st.dataframe(df)

    if st.form_submit_button('Cadastrar'):
        if not nome or df is None:
            st.warning('Complete o cadastro')
        else:
            if "age" not in df.columns or "sex" not in df.columns:
                st.error("O CSV precisa ter as colunas: Age e Sex")
            else:
                idade = int(df.iloc[0]["age"])
                sex_val = int(df.iloc[0]["sex"])
                sexo = "Homem" if sex_val == 1 else "Mulher" if sex_val == 0 else "Não informado"

                # Ajuste este valor quando tiver o resultado do modelo
                resultado = "Pendente"

                pac_con.db_insert(nome, idade, sexo, resultado)
                with st.spinner("Cadastrando..."):
                    time.sleep(2)
                st.success("Paciente cadastrado com sucesso!")
                time.sleep(2)
                st.rerun()



with st.container(border=True):
    st.subheader("Pacientes cadastrados")
    pacientes = pac_con.selecionar_todos()
    st.dataframe([p.to_dict() for p in pacientes])
    if pacientes:
        ids = [p.id for p in pacientes]
        id_to_delete = st.selectbox("Selecione o ID do paciente para excluir", ids)
        if st.button("Excluir paciente"):
            pac_con.excluir(id_to_delete)
            st.success(f"Paciente com ID {id_to_delete} excluído com sucesso!")
            st.rerun()

