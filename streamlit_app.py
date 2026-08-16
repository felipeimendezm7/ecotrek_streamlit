import streamlit as st
import pandas as pd
import plotly.express as px
import nltk
import matplotlib.pyplot as plt

from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud, STOPWORDS

# ------------------------------------------------------------
# Configuración de página
# ------------------------------------------------------------
st.set_page_config(
    page_title="EcoTrek Solutions",
    page_icon="🌿",
    layout="wide"
)

st.title("🌿 EcoTrek Solutions")
st.subheader("Análisis de Sentimiento de Reseñas de Clientes")

# ------------------------------------------------------------
# Carga del archivo
# ------------------------------------------------------------
archivo = st.file_uploader(
    "Sube el archivo ecosmart_reviews.csv",
    type=["csv"]
)

if archivo is not None:

    df_reviews = pd.read_csv(archivo)

    st.success("Archivo cargado correctamente")

    # ------------------------------------------------------------
    # Configuración de VADER
    # ------------------------------------------------------------
    nltk.download("vader_lexicon", quiet=True)
    analyzer = SentimentIntensityAnalyzer()

    # ------------------------------------------------------------
    # Análisis de sentimiento
    # ------------------------------------------------------------
    def clasificar_sentimiento(texto):
        compound = analyzer.polarity_scores(str(texto))["compound"]

        if compound >= 0.05:
            return "Positivo"
        elif compound <= -0.05:
            return "Negativo"
        else:
            return "Neutral"

    df_reviews["Sentimiento"] = df_reviews["Reseña"].apply(
        clasificar_sentimiento
    )

    # ------------------------------------------------------------
    # Conteo y porcentajes
    # ------------------------------------------------------------
    df_conteo = (
        df_reviews["Sentimiento"]
        .value_counts()
        .reset_index()
    )

    df_conteo.columns = ["Sentimiento", "Conteo"]

    df_conteo["Porcentaje"] = (
        df_conteo["Conteo"]
        / df_conteo["Conteo"].sum()
        * 100
    ).round(1)

    st.subheader("Distribución de sentimientos")

    st.dataframe(df_conteo, use_container_width=True)

    # ------------------------------------------------------------
    # Gráfico circular
    # ------------------------------------------------------------
    fig_pie = px.pie(
        df_conteo,
        values="Conteo",
        names="Sentimiento",
        title="Distribución de Sentimientos de Clientes"
    )

    fig_pie.update_layout(title_x=0.5)

    st.plotly_chart(
        fig_pie,
        use_container_width=True
    )

    # ------------------------------------------------------------
    # Función para generar WordCloud
    # ------------------------------------------------------------
    def generar_wordcloud(
        df,
        sentimiento_filtro,
        background_color,
        colormap
    ):

        texto_filtrado = " ".join(
            df[
                df["Sentimiento"] == sentimiento_filtro
            ]["Reseña"].astype(str)
        )

        stopwords_en = set(STOPWORDS)

        stopwords_en.update([
            "the", "a", "an", "is", "it", "its",
            "of", "and", "or", "to", "in", "on",
            "for", "with", "this", "that", "i",
            "you", "he", "she", "we", "they",
            "not", "but", "have", "has", "had",
            "do", "does", "did", "would", "will",
            "can", "just", "don", "t", "s", "m",
            "re", "ll", "ve", "about", "out",
            "up", "down", "get", "much", "many",
            "good", "great", "really", "very",
            "my", "your", "so", "from", "also"
        ])

        wordcloud = WordCloud(
            width=900,
            height=450,
            background_color=background_color,
            colormap=colormap,
            stopwords=stopwords_en,
            min_font_size=10,
            max_words=80,
            collocations=False
        ).generate(texto_filtrado)

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.imshow(
            wordcloud,
            interpolation="bilinear"
        )

        ax.axis("off")

        ax.set_title(
            f"Nube de Palabras - {sentimiento_filtro}",
            fontsize=16
        )

        return fig

    # ------------------------------------------------------------
    # Nubes de palabras
    # ------------------------------------------------------------
    st.subheader("Nubes de palabras")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("### Positivas")
        fig_pos = generar_wordcloud(
            df_reviews,
            "Positivo",
            "white",
            "Greens"
        )
        st.pyplot(fig_pos)

    with col2:
        st.write("### Neutrales")
        fig_neu = generar_wordcloud(
            df_reviews,
            "Neutral",
            "white",
            "Blues"
        )
        st.pyplot(fig_neu)

    with col3:
        st.write("### Negativas")
        fig_neg = generar_wordcloud(
            df_reviews,
            "Negativo",
            "white",
            "Reds"
        )
        st.pyplot(fig_neg)
