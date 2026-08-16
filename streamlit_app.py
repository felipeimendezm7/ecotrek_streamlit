import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="EcoTrek Solutions",
    page_icon="🌿",
    layout="wide"
)

st.title("🌿 EcoTrek Solutions")
st.write("Análisis de reseñas de clientes")

archivo = st.file_uploader(
    "Sube el archivo ecosmart_reviews.csv",
    type=["csv"]
)

if archivo is not None:
    df_reviews = pd.read_csv(archivo)

    st.success("Archivo cargado correctamente")

    st.subheader("Vista previa de los datos")
    st.dataframe(df_reviews)
