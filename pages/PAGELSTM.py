import streamlit as st

btn1,btn2 = st.columns(2, gap="medium",border=False, width=400,vertical_alignment='bottom')
with btn1:
    if st.button("Página inicial"):
        st.switch_page("main.py")

with btn2:
    if st.button("Modelo RealMLP"):
        st.switch_page("pages/PAGEMLP.py")
st.divider()

st.markdown("# Modelo escolhido: RealMLP")
st.markdown("""###  Informações relevantes sobre o modelo escolhido na aba atual: 
            As redes Long Short-Term Memory (LSTMs) constituem uma classe de Redes Neurais Recorrentes (RNNs) capazes de capturar e aprender dependências temporais de longo alcance. 
            fundamentam-se em células de memória e um conjunto de mecanismos de controle, denominados 'gates', que regulam o fluxo de informação. Através dos gates de entrada, 
            esquecimento e saída, a LSTM adquire a capacidade de selecionar quais informações devem ser armazenadas, atualizadas ou descartadas ao longo do tempo. Este controle 
            seletivo permite a retenção de dependências temporais prolongadas, tornando a LSTM uma arquitetura de rede neural recorrente particularmente adequada para tarefas como 
            processamento de linguagem natural, previsão de séries temporais e reconhecimento de padrões em dados sequenciais.""")