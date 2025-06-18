import streamlit as st
import pandas as pd
from transformers import pipeline
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

# Page config
st.set_page_config(layout="wide")
st.title("📊 Dashboard Analisis Sentimen Ulasan Produk")

# Load sentiment pipeline
sent_pipeline = pipeline("sentiment-analysis")

# Sidebar for file upload
st.sidebar.header("🔽 Upload Dataset CSV")
uploaded_file = st.sidebar.file_uploader("Upload file CSV", type=["csv"])

df = None
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        df = df.head(1000)  # limit for performance
        st.success(f"✅ Data berhasil dimuat: {df.shape[0]} baris, {df.shape[1]} kolom.")
    except Exception as e:
        st.error(f"Gagal memuat file: {e}")

if df is not None:
    st.subheader("📄 Data Sample (Top 1000)")
    st.dataframe(df)

    review_col = st.selectbox("📌 Pilih kolom teks ulasan:", df.columns)

    if st.button("🔍 Analisis Sentimen Semua"):
        with st.spinner("Menganalisis seluruh ulasan..."):
            df.drop(columns=['Sentiment', 'Score'], errors='ignore', inplace=True)

            sentiments = df[review_col].astype(str).apply(lambda x: sent_pipeline(x)[0])
            df['Sentiment'] = sentiments.apply(lambda x: x['label'])
            df['Score'] = sentiments.apply(lambda x: x['score'])

            st.success("Analisis sentimen selesai!")

            st.subheader("Hasil Analisis Sentimen")
            st.dataframe(df[[review_col, 'Sentiment', 'Score']])

            st.subheader("Distribusi Sentimen")
            sentiment_counts = df['Sentiment'].value_counts()
            st.bar_chart(sentiment_counts)

            score_avg = df.groupby('Sentiment')['Score'].mean().reset_index()
            fig = go.Figure(go.Bar(
                x=score_avg['Score'],
                y=score_avg['Sentiment'],
                orientation='h',
                marker_color='orange',
                text=score_avg['Score'].round(3),
                textposition="auto"
            ))
            st.subheader("Rata-rata Skor per Sentimen")
            fig.update_layout(xaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)

            # Advanced visualizations
            st.subheader("Count of Reviews by Stars")
            st.image("image/count_review.png", caption="Count of Reviews by Stars")

            st.subheader("Compund Score by Amazon Star Review")
            st.image("image/compund_review.png", caption="Compund Score by Amazon Star Review")
            st.image("image/analisis_sentiment_score.png", caption="Analisis Sentiment Score")

            st.subheader("Pairplot Score Sentiment")
            st.image("image/pairplot_score.png", caption="Pairplot Score Sentiment")

    # Manual analysis
    st.subheader("Uji Sentimen Manual")
    text_input = st.text_area("Masukkan kalimat untuk dianalisis:")

    if st.button("🔍 Analisis Kalimat"):
        if text_input.strip():
            with st.spinner("Menganalisis..."):
                result = sent_pipeline(text_input)[0]
                st.markdown(f"**Label Sentimen:** `{result['label']}`")
                st.markdown(f"**Skor Keyakinan:** `{result['score']:.4f}`")

                fig = go.Figure(go.Bar(
                    x=[result['score']],
                    y=[result['label']],
                    orientation='h',
                    marker_color='skyblue',
                    text=[f"{result['score']:.4f}"],
                    textposition="auto"
                ))
                fig.update_layout(title="Skor Sentimen", xaxis_range=[0, 1])
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("❗ Masukkan kalimat terlebih dahulu.")

   
else:
    st.info("📤 Silakan upload file CSV terlebih dahulu untuk memulai.")

