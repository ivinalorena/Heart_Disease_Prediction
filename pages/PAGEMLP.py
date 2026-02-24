import streamlit as st
import time
import os
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from Controllers import paciente_controller as pac_con

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

MODEL_PATH = Path("realmlp_td_classifier_model.pkl")
UPLOADS_DIR = Path("data/uploads")
REFERENCE_DATA_PATH = Path("data/Heart_Disease_Prediction.csv")
EXPECTED_FEATURES = [
    "Age", "Sex", "Chest pain type", "BP", "Cholesterol", "FBS over 120",
    "EKG results", "Max HR", "Exercise angina", "ST depression",
    "Slope of ST", "Number of vessels fluro", "Thallium"
]

def _normalize(col_name: str) -> str:
    return col_name.strip().lower().replace(" ", "_")


EXPECTED_MAP = {_normalize(col): col for col in EXPECTED_FEATURES}


def _force_cpu_recursively(obj, visited=None, depth=0):
    if obj is None:
        return
    if visited is None:
        visited = set()
    if depth > 4:
        return

    obj_id = id(obj)
    if obj_id in visited:
        return
    visited.add(obj_id)

    if hasattr(obj, "set_params"):
        try:
            obj.set_params(device="cpu")
        except Exception:
            pass

    for attr_name in ("device", "_device"):
        if hasattr(obj, attr_name):
            try:
                setattr(obj, attr_name, "cpu")
            except Exception:
                pass

    if isinstance(obj, dict):
        for value in obj.values():
            _force_cpu_recursively(value, visited, depth + 1)
        return

    if isinstance(obj, (list, tuple, set)):
        for value in obj:
            _force_cpu_recursively(value, visited, depth + 1)
        return

    if hasattr(obj, "__dict__"):
        for value in obj.__dict__.values():
            _force_cpu_recursively(value, visited, depth + 1)


@st.cache_resource
def load_realmlp_model(model_path: Path):
    if not model_path.exists():
        return None
    model = joblib.load(model_path)
    _force_cpu_recursively(model)
    return model


@st.cache_data
def load_reference_data(reference_path: Path) -> pd.DataFrame:
    if not reference_path.exists():
        raise FileNotFoundError("Dataset de referência não encontrado para criar features engenheiradas.")

    reference_df = pd.read_csv(reference_path, header=0, index_col=False)
    if "Heart Disease" not in reference_df.columns:
        raise ValueError("O dataset de referência precisa ter a coluna 'Heart Disease'.")

    encoder = LabelEncoder()
    reference_df["Heart Disease"] = encoder.fit_transform(reference_df["Heart Disease"])
    return reference_df


def add_engineered_features(df_base: pd.DataFrame, reference_df: pd.DataFrame) -> pd.DataFrame:
    df_temp = df_base.copy()

    for col in EXPECTED_FEATURES:
        stats = reference_df.groupby(col)["Heart Disease"].agg(["mean", "median", "std", "skew", "count"]).reset_index()
        stats.columns = [col] + [f"orig_{col}_{s}" for s in ["mean", "median", "std", "skew", "count"]]

        df_temp = df_temp.merge(stats, on=col, how="left")

        fill_values = {
            f"orig_{col}_mean": reference_df["Heart Disease"].mean(),
            f"orig_{col}_median": reference_df["Heart Disease"].median(),
            f"orig_{col}_std": 0,
            f"orig_{col}_skew": 0,
            f"orig_{col}_count": 0,
        }
        df_temp = df_temp.fillna(value=fill_values)

    return df_temp


def prepare_model_input(df_uploaded: pd.DataFrame) -> pd.DataFrame:
    df_norm = df_uploaded.copy()
    df_norm.columns = [_normalize(c) for c in df_norm.columns]

    missing = [norm_col for norm_col in EXPECTED_MAP.keys() if norm_col not in df_norm.columns]
    if missing:
        missing_raw = [EXPECTED_MAP[m] for m in missing]
        raise ValueError(f"CSV sem colunas obrigatórias: {', '.join(missing_raw)}")

    rename_back_to_raw = {norm_col: raw_col for norm_col, raw_col in EXPECTED_MAP.items()}
    df_model = df_norm.rename(columns=rename_back_to_raw)
    df_model = df_model[EXPECTED_FEATURES].copy()

    reference_df = load_reference_data(REFERENCE_DATA_PATH)
    df_model = add_engineered_features(df_model, reference_df)

    expected_full_order = EXPECTED_FEATURES.copy()
    for col in EXPECTED_FEATURES:
        expected_full_order.extend([f"orig_{col}_{s}" for s in ["mean", "median", "std", "skew", "count"]])
    df_model = df_model[expected_full_order]

    for col in df_model.columns:
        df_model[col] = df_model[col].astype(str).astype("category")

    return df_model


def save_uploaded_dataframe(df_uploaded: pd.DataFrame, nome: str) -> Path:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(ch for ch in nome if ch.isalnum() or ch in ("_", "-", " ")).strip().replace(" ", "_")
    safe_name = safe_name or "paciente"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = UPLOADS_DIR / f"{safe_name}_{timestamp}.csv"
    df_uploaded.to_csv(file_path, index=False)
    return file_path

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

    loaded_model = load_realmlp_model(MODEL_PATH)
    if loaded_model is None:
        st.error("Modelo não encontrado: realmlp_td_classifier_model.pkl")

    df = None
    if arquivo is not None:
        df = pd.read_csv(arquivo,header=0,index_col=False)
        st.dataframe(df)

    if st.form_submit_button('Cadastrar'):
        if not nome or df is None:
            st.warning('Complete o cadastro')
        else:
            if loaded_model is None:
                st.error("Não foi possível prever sem o arquivo do modelo treinado.")
            else:
                try:
                    df_model = prepare_model_input(df)
                except (ValueError, FileNotFoundError) as err:
                    st.error(str(err))
                    st.stop()

                idade = int(float(df_model.iloc[0]["Age"]))
                sex_val = int(float(df_model.iloc[0]["Sex"]))
                sexo = "Homem" if sex_val == 1 else "Mulher" if sex_val == 0 else "Não informado"

                try:
                    prediction_probability = float(loaded_model.predict_proba(df_model)[0][1])
                except AssertionError as err:
                    if "Torch not compiled with CUDA enabled" in str(err):
                        _force_cpu_recursively(loaded_model)
                        prediction_probability = float(loaded_model.predict_proba(df_model)[0][1])
                    else:
                        raise
                predicted_class = 1 if prediction_probability >= 0.5 else 0
                resultado = "Presença de doença Cardíaca" if predicted_class == 1 else "Ausência de doença Cardíaca"
                resultado_db = f"{resultado} ({prediction_probability:.2%})"

                saved_file = save_uploaded_dataframe(df, nome)

                pac_con.db_insert(nome, idade, sexo, resultado_db)
                with st.spinner("Cadastrando..."):
                    time.sleep(2)
                st.success("Paciente cadastrado com sucesso!")
                #st.info(f"Predição: {resultado_db}")
                #st.caption(f"Arquivo salvo em: {saved_file}")
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

