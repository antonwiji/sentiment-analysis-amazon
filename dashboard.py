import streamlit as st
import pandas as pd
import numpy as np
from transformers import pipeline
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import re
from wordcloud import WordCloud

import torch
from matplotlib import colormaps

# --- Bagian 1: Konfigurasi Aplikasi Streamlit ---
# ------------------------------------------------
st.set_page_config(layout="wide")
st.title("📊 Dashboard Analisis Perilaku Pelanggan & Rekomendasi Produk")

st.sidebar.header("🎯 Tujuan Analisis")
st.sidebar.markdown("""
- Menemukan pola perilaku pelanggan
- Melakukan analisis sentimen ulasan
- Menyediakan sistem rekomendasi sederhana
- Klasterisasi pelanggan
""")

# --- CSS Kustom untuk Ukuran Font Tab dan Pembatas ---
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.25rem;
        font-weight: bold;
    }

    .stTabs [data-baseweb="tab-list"] {
        border-bottom: 2px solid #6c757d;
        margin-bottom: 1rem;
    }

    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        border-bottom: 3px solid #007bff;
    }
</style>
""", unsafe_allow_html=True)


# --- Bagian 2: Fungsi Pembantu (Helper Functions) ---
# ----------------------------------------------------

def make_arrow_safe(df_input):
    """
    Fungsi pembantu untuk mengonversi kolom objek (string) dalam DataFrame
    menjadi tipe 'str' agar kompatibel dengan Apache Arrow (optimizer Streamlit).
    """
    df = df_input.copy()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str)
    return df

# Fungsi untuk menentukan warna berdasarkan rating
def get_rating_color(rating):
    if rating >= 4.0:
        return 'lightgreen' # Hijau untuk rating bagus
    elif rating >= 3.0:
        return 'gold'       # Kuning untuk rating sedang
    else:
        return 'salmon'     # Merah untuk rating rendah

# --- Bagian 3: Inisialisasi State Aplikasi (Session State) ---
# -------------------------------------------------------------
if 'df_sentiment' not in st.session_state:
    st.session_state.df_sentiment = None
if 'sentiment_metrics' not in st.session_state:
    st.session_state.sentiment_metrics = {}
if 'uploaded_file_name' not in st.session_state:
    st.session_state.uploaded_file_name = None
if 'top_n_rekom' not in st.session_state:
    st.session_state.top_n_rekom = 5
if 'min_reviews_rekom' not in st.session_state:
    st.session_state.min_reviews_rekom = 10
if 'sort_option_rekom' not in st.session_state:
    st.session_state.sort_option_rekom = "AvgRating"

# --- Bagian 4: Unggah Data CSV ---
# ---------------------------------
st.sidebar.header("📥 Upload Dataset CSV")
uploaded_file = st.sidebar.file_uploader("Upload file CSV", type=["csv"])

# --- PERBAIKAN DI SINI: Pindahkan slider ke luar blok if uploaded_file ---
# Ini memastikan slider selalu ada di sidebar setelah file uploader.
# row_limit = st.sidebar.slider("🔢 Jumlah data yang ingin dimuat (max)", min_value=100, max_value=500000, value=1000, step=500, key="row_limit_data_slider")
# Saya juga menambahkan 'key' unik untuk stabilitas widget.
# --- AKHIR PERBAIKAN ---

if uploaded_file:
    with st.status("Memuat dan memproses data, harap tunggu...", expanded=True) as status:
        df_full = pd.read_csv(uploaded_file)
        max_rows = len(df_full)

        row_limit = st.sidebar.slider(
            "🔢 Jumlah data yang ingin dimuat",
            min_value=100,
            max_value=max_rows,
            value=min(1000, max_rows),
            step=100,
            key="row_limit_data_slider"
        )

        df = df_full.head(row_limit)

        # Check if the file has changed or if df_sentiment is not yet loaded
        if st.session_state.df_sentiment is not None and st.session_state.uploaded_file_name == uploaded_file.name:
            st.write("Menggunakan data yang sudah dimuat sebelumnya.")
            df = st.session_state.df_sentiment.copy()
        else:
            st.write("Membaca file CSV...")
            # Gunakan nilai row_limit yang sudah didefinisikan di atas
            # df = pd.read_csv(uploaded_file).head(row_limit)
            st.write("Mengkonversi tipe data...")
            df = df.convert_dtypes()
            for col in df.select_dtypes(include=['object']).columns:
                df[col] = df[col].astype(str)
            
            st.session_state.df_sentiment = None # Reset df_sentiment, will be populated after sentiment analysis
            st.session_state.sentiment_metrics = {}
            st.session_state.uploaded_file_name = uploaded_file.name
        
        status.update(label="✅ Data berhasil dimuat dan siap!", state="complete", expanded=False)
    
    st.success(f"✅ Data berhasil dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Eksplorasi Data (EDA)", "🎯 Rekomendasi", "💬 Sentimen", "👥 Klasterisasi", "📝 Kesimpulan"])

    # --- Tab 1: Eksplorasi Data (EDA) ---
    with tab1:
        st.subheader("🔍 Eksplorasi Data (EDA)")
        st.markdown("---") 

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Distribusi Rating**")
            st.markdown("Diagram ini menunjukkan sebaran jumlah rating produk dari pengguna.")
            if "Score" in df.columns:
                st.bar_chart(df["Score"].value_counts().sort_index())

        with col2:
            st.markdown("**Produk Paling Populer**")
            if "ProductId" in df.columns:
                popular = df["ProductId"].value_counts().head(10)
                st.markdown("Diagram ini menampilkan 10 produk dengan jumlah ulasan terbanyak.")
                st.bar_chart(popular)

        if "UserId" in df.columns:
            st.markdown("**Reviewer Aktif**")
            st.caption("⚠️ Statistik ini berdasarkan maksimum data yang dimuat: " + str(df.shape[0]) + " baris pertama.")
            active = df["UserId"].value_counts().head(10).reset_index()
            active.columns = ["UserId", "Jumlah Review"]
            st.dataframe(make_arrow_safe(active))

        st.markdown("### Statistik Deskriptif (Kategorikal)")
        st.caption("⚠️ Statistik ini mencerminkan kolom kategorikal dari maksimum " + str(df.shape[0]) + " baris data.")
        categorical_cols = df.select_dtypes(include=['object', 'string']).columns
        categorical_desc = df[categorical_cols].describe(include='all').T
        st.dataframe(make_arrow_safe(categorical_desc))

        if "ProductId" in df.columns:
            st.subheader("🌀 Wordcloud Produk Terpopuler")
            st.markdown("Visualisasi ini menunjukkan kata-kata yang paling sering muncul pada Product ID.")
            product_text = " ".join(df['ProductId'].astype(str))
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(product_text)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)

        # --- Tambahan Diagram Baru di EDA ---
        # 1. Tren Rating Produk dari Waktu ke Waktu
        st.subheader("📈 Tren Rating Produk dari Waktu ke Waktu")
        st.markdown("Visualisasi ini menunjukkan bagaimana rata-rata rating produk berubah sepanjang waktu, memberikan insight tentang performa produk secara historis.")
        
        data_for_eda_time_trend = df.copy()

        if 'Time' in data_for_eda_time_trend.columns and 'Score' in data_for_eda_time_trend.columns:
            try:
                data_for_eda_time_trend['Datetime'] = pd.to_datetime(data_for_eda_time_trend['Time'], unit='s', errors='coerce')
                data_for_eda_time_trend.dropna(subset=['Datetime'], inplace=True)

                if not data_for_eda_time_trend.empty:
                    data_for_eda_time_trend['Year'] = data_for_eda_time_trend['Datetime'].dt.year
                    avg_rating_yearly = data_for_eda_time_trend.groupby('Year')['Score'].mean().reset_index()

                    fig_rating_trend = go.Figure(data=go.Scatter(x=avg_rating_yearly['Year'], y=avg_rating_yearly['Score'], mode='lines+markers', name='Rata-rata Rating'))
                    fig_rating_trend.update_layout(title='Rata-rata Rating Produk per Tahun',
                                                   xaxis_title='Tahun',
                                                   yaxis_title='Rata-rata Rating',
                                                   hovermode="x unified")
                    st.plotly_chart(fig_rating_trend)
                    
                    st.markdown("""
                    **Analisis Hasil:**
                    * **Tren Naik/Turun:** Amati apakah ada pola peningkatan atau penurunan rata-rata rating dari tahun ke tahun. Peningkatan bisa menunjukkan peningkatan kualitas produk atau kepuasan pelanggan, sementara penurunan bisa menjadi sinyal adanya masalah yang perlu diselidiki.
                    * **Stabilitas Rating:** Jika garis relatif datar, ini menunjukkan rating produk cenderung stabil dari waktu ke waktu.
                    * **Anomali:** Perhatikan lonjakan atau atau penurunan drastis pada tahun tertentu. Ini bisa terkait dengan peluncuran produk baru, kampanye pemasaran, atau isu kualitas yang terjadi pada periode tersebut.
                    """)
                else:
                    st.warning("Data tanggal tidak cukup atau tidak valid untuk membuat tren rating.")
            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses kolom 'Time' untuk tren rating: {e}")
                st.info("Pastikan kolom 'Time' Anda dalam format Unix timestamp (detik) atau format tanggal yang dikenali Pandas.")
        else:
            st.warning("Kolom 'Time' atau 'Score' tidak ditemukan dalam dataset. Tren rating tidak dapat ditampilkan.")


    # --- Tab 2 (Baru): Rekomendasi Produk Sederhana ---
    with tab2:
        st.subheader("🎯 Rekomendasi Produk Sederhana")
        st.markdown("---")

        if "ProductId" in df.columns and "Score" in df.columns:
            st.session_state.top_n_rekom = st.slider("Berapa produk tertinggi yang ingin ditampilkan?", 1, 20, value=st.session_state.top_n_rekom, key="slider_top_n")
            st.session_state.min_reviews_rekom = st.slider("Minimum jumlah review untuk direkomendasikan", 1, 100, value=st.session_state.min_reviews_rekom, key="slider_min_reviews")
            
            rekom_df = df.groupby("ProductId")["Score"].agg(AvgRating="mean", NumReviews="count")
            rekom_df = rekom_df[rekom_df["NumReviews"] >= st.session_state.min_reviews_rekom].sort_values(by="AvgRating", ascending=False)
            
            st.session_state.sort_option_rekom = st.radio("Urutkan berdasarkan:", ["AvgRating", "NumReviews"], horizontal=True, key="radio_sort_rekom", index=0 if st.session_state.sort_option_rekom == "AvgRating" else 1)
            rekom_df = rekom_df.sort_values(by=st.session_state.sort_option_rekom, ascending=False).head(st.session_state.top_n_rekom)
            
            st.dataframe(make_arrow_safe(rekom_df.reset_index()))

            st.subheader("📌 Visualisasi Produk Terbaik Berdasarkan Rating")
            if not rekom_df.empty:
                # Membuat daftar warna dinamis untuk rekomendasi
                rekom_colors = [get_rating_color(rating) for rating in rekom_df['AvgRating']]
                
                # Membuat teks hover kustom untuk visualisasi rekomendasi
                rekom_hover_texts = [
                    f"Produk: {row.name}<br>"
                    f"Jumlah Ulasan: {row['NumReviews']}<br>"
                    f"Rata-rata Rating: {row['AvgRating']:.2f}"
                    for index, row in rekom_df.iterrows()
                ]


                fig = go.Figure(go.Bar(
                    x=rekom_df["AvgRating"],
                    y=rekom_df.index,
                    orientation='h',
                    marker_color=rekom_colors, # Menggunakan daftar warna dinamis di sini
                    text=rekom_df["AvgRating"].round(2),
                    textposition='auto',
                    hoverinfo='text', # Aktifkan hover info sebagai teks kustom
                    hovertext=rekom_hover_texts # Gunakan teks kustom yang sudah dibuat
                ))
                fig.update_layout(title="Top Produk Rekomendasi", xaxis_title="Rating Rata-rata", xaxis_range=[0,5]) # Menyesuaikan rentang X-axis
                st.markdown("Visualisasi ini membantu memahami produk dengan rating tertinggi berdasarkan nilai rata-rata.")
                st.plotly_chart(fig)
                st.markdown("""
                **Keterangan Warna Rata-rata Rating:**
                * **Hijau Muda (`lightgreen`):** Rating $4.0$ ke atas (Sangat Baik)
                * **Kuning (`gold`):** Rating $3.0$ sampai kurang dari $4.0$ (Cukup Baik)
                * **Merah Salmon (`salmon`):** Rating kurang dari $3.0$ (Perlu Perhatian)
                """)
            else:
                st.info("Tidak ada produk yang memenuhi kriteria rekomendasi.")

            st.subheader("📌 Insight Rekomendasi Produk")
            
            if not rekom_df.empty:
                good = rekom_df[rekom_df['AvgRating'] >= 4.5]
                improve = rekom_df[(rekom_df['AvgRating'] >= 3.0) & (rekom_df['AvgRating'] < 4.0)]
                drop = rekom_df[rekom_df['AvgRating'] < 3.0]
            else:
                good = pd.DataFrame(columns=['AvgRating', 'NumReviews'])
                improve = pd.DataFrame(columns=['AvgRating', 'NumReviews'])
                drop = pd.DataFrame(columns=['AvgRating', 'NumReviews'])

            st.markdown("✅ **Produk yang Layak Diteruskan (Rating ≥ 4.5):**")
            if not good.empty:
                st.dataframe(make_arrow_safe(good))
            else:
                st.info("Tidak ada produk yang layak diteruskan berdasarkan kriteria saat ini.")

            st.markdown("🛠️ **Produk yang Perlu Ditingkatkan (3.0 ≤ Rating < 4.0):**")
            if not improve.empty:
                st.dataframe(make_arrow_safe(improve))
            else:
                st.info("Tidak ada produk yang perlu ditingkatkan berdasarkan kriteria saat ini.")

            st.markdown("⚠️ **Produk Potensial untuk Dihentikan (Rating < 3.0):**")
            if not drop.empty:
                st.dataframe(make_arrow_safe(drop))
            else:
                st.info("Tidak ada produk potensial untuk dihentikan berdasarkan kriteria saat ini.")

        else:
            st.warning("Kolom 'ProductId' atau 'Score' tidak ditemukan dalam dataset. Tidak dapat melakukan rekomendasi.")

    # --- Tab 3 (Baru): Analisis Sentimen ---
    with tab3:
        st.markdown("""
---
## 🔍 Pemodelan Sentimen
Silakan jalankan analisis sentimen ulasan pelanggan untuk mendapatkan insight lebih lanjut.
""")
        st.subheader("💬 Analisis Sentimen dengan Transformers")
        st.markdown("---") 

        if torch.cuda.is_available():
            device_info = "Menggunakan GPU (CUDA)."
            device = 0
        else:
            device_info = "Menggunakan CPU."
            device = -1
        st.info(f"Status perangkat untuk analisis sentimen: {device_info}")

        sent_pipeline = pipeline("sentiment-analysis")

        if st.button("🔍 Jalankan Analisis Sentimen"):
            with st.spinner("Sedang memproses..."):
                sentiment_labels = []
                sentiment_scores = []
                import time
                start_time = time.time()
                
                progress = st.progress(0, text="🔄 Memproses sentimen...")
                texts = df['Text'].astype(str).tolist()
                batch_size = 128
                
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i+batch_size]
                    batch = [text[:512] for text in batch]
                    results = sent_pipeline(batch)
                    
                    for res in results:
                        sentiment_labels.append(res['label'])
                        sentiment_scores.append(res['score'])
                    
                    percent_complete = min((i + batch_size) / len(texts), 1.0)
                    est_total = (time.time() - start_time) / percent_complete
                    est_remaining = est_total - (time.time() - start_time)
                    progress.progress(percent_complete, text=f"⏳ {est_remaining:.1f} detik tersisa")
                
                elapsed = time.time() - start_time
                st.success(f"⏱️ Waktu proses: {elapsed:.2f} detik")
                
                df['Sentiment'] = sentiment_labels
                df['SentimentScore'] = sentiment_scores
            
            st.session_state.df_sentiment = df.copy()
            st.success("✅ Analisis sentimen selesai!")

            from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, f1_score, roc_auc_score

            if 'Score' in df.columns and df['Sentiment'].isin(['POSITIVE', 'NEGATIVE']).all():
                df['SentimentTrue'] = df['Score'].apply(lambda x: 'POSITIVE' if x >= 3 else 'NEGATIVE') 
                y_true = df['SentimentTrue']
                y_pred = df['Sentiment']
                
                if len(y_true) > 0 and len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1:
                    
                    accuracy = accuracy_score(y_true, y_pred)
                    precision = precision_score(y_true, y_pred, pos_label='POSITIVE', zero_division=0)
                    f1 = f1_score(y_true, y_pred, pos_label='POSITIVE', zero_division=0)
                    
                    try:
                        roc_auc = roc_auc_score(y_true.map({'NEGATIVE': 0, 'POSITIVE': 1}), y_pred.map({'NEGATIVE': 0, 'POSITIVE': 1}))
                    except ValueError:
                        roc_auc = 'N/A (single class)'

                    st.subheader("📊 Evaluasi Model Sentimen")
                    st.markdown("**📋 Metrik Tambahan:**")
                    st.write({
                        'Accuracy': f"{accuracy:.2f}",
                        'Precision': f"{precision:.2f}",
                        'F1-Score': f"{f1:.2f}",
                        'ROC AUC': f"{roc_auc}"
                    })
                    
                    st.session_state.sentiment_metrics = {
                        'accuracy': accuracy,
                        'precision': precision,
                        'f1': f1,
                        'roc_auc': roc_auc
                    }

                    st.subheader("📌 Confusion Matrix")
                    cm = confusion_matrix(df['SentimentTrue'], df['Sentiment'], labels=['NEGATIVE', 'POSITIVE'])
                    fig, ax = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['NEGATIVE', 'POSITIVE'], yticklabels=['NEGATIVE', 'POSITIVE'])
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    st.pyplot(fig)
                else:
                    st.warning("Tidak cukup variasi label sentimen atau rating untuk menghitung metrik evaluasi model.")
                    st.session_state.sentiment_metrics = {}
            else:
                st.warning("Kolom 'Score' tidak ditemukan atau sentimen yang terdeteksi bukan hanya POSITIVE/NEGATIVE untuk evaluasi model.")


        if st.session_state.df_sentiment is not None and 'Sentiment' in st.session_state.df_sentiment.columns:
            df = st.session_state.df_sentiment.copy()
            preview_df = df[["Text", "Sentiment", "SentimentScore"]].copy()
            for col in preview_df.columns:
                if preview_df[col].dtype == 'object':
                    preview_df[col] = preview_df[col].astype(str)
            st.subheader("📝 Preview Hasil Analisis Sentimen")
            st.dataframe(preview_df)

            st.subheader("📈 Distribusi Sentimen")
            st.markdown("Visualisasi ini menunjukkan jumlah ulasan berdasarkan label sentimen yang terdeteksi.")
            st.bar_chart(df["Sentiment"].value_counts())

            st.subheader("📊 Rata-rata Skor Sentimen")
            avg_score = df.groupby("Sentiment")["SentimentScore"].mean().reset_index()
            fig = go.Figure(go.Bar(
                x=avg_score["SentimentScore"],
                y=avg_score["Sentiment"],
                orientation='h',
                marker_color='green'
            ))
            st.markdown("Diagram batang horizontal ini menampilkan rata-rata confidence score untuk tiap label sentimen.")
            st.plotly_chart(fig)

            st.subheader("📈 Tren Sentimen dari Tahun ke Tahun")
            if 'Time' in df.columns:
                try:
                    df_trend = df.copy()
                    df_trend['Datetime'] = pd.to_datetime(df_trend['Time'], unit='s', errors='coerce')
                    df_trend.dropna(subset=['Datetime'], inplace=True)

                    if not df_trend.empty:
                        df_trend['Year'] = df_trend['Datetime'].dt.year
                        
                        sentiment_yearly = df_trend.groupby(['Year', 'Sentiment']).size().unstack(fill_value=0)
                        sentiment_yearly_pct = sentiment_yearly.divide(sentiment_yearly.sum(axis=1), axis=0) * 100

                        fig_trend = go.Figure()
                        if 'POSITIVE' in sentiment_yearly_pct.columns:
                            fig_trend.add_trace(go.Scatter(x=sentiment_yearly_pct.index, y=sentiment_yearly_pct['POSITIVE'],
                                                            mode='lines+markers', name='Positive Sentiment',
                                                            line=dict(color='lightgreen')))
                        if 'NEGATIVE' in sentiment_yearly_pct.columns:
                            fig_trend.add_trace(go.Scatter(x=sentiment_yearly_pct.index, y=sentiment_yearly_pct['NEGATIVE'],
                                                            mode='lines+markers', name='Negative Sentiment',
                                                            line=dict(color='salmon')))
                        if 'NEUTRAL' in sentiment_yearly_pct.columns:
                            fig_trend.add_trace(go.Scatter(x=sentiment_yearly_pct.index, y=sentiment_yearly_pct['NEUTRAL'],
                                                            mode='lines+markers', name='Neutral Sentiment',
                                                            line=dict(color='lightgrey')))
                        
                        fig_trend.update_layout(title='Tren Persentase Sentimen per Tahun',
                                                xaxis_title='Tahun',
                                                yaxis_title='Persentase Sentimen (%)',
                                                hovermode="x unified")
                        st.plotly_chart(fig_trend)
                    else:
                        st.warning("Data tanggal tidak cukup atau tidak valid untuk membuat tren sentimen.")
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat memproses kolom 'Time' untuk tren sentimen: {e}")
                    st.info("Pastikan kolom 'Time' Anda dalam format Unix timestamp (detik) atau format tanggal yang dikenali Pandas.")
            else:
                st.warning("Kolom 'Time' tidak ditemukan dalam dataset. Tren sentimen dari tahun ke tahun tidak dapat ditampilkan.")
        else:
            st.info("Klik 'Jalankan Analisis Sentimen' untuk melihat hasil dan visualisasi.")

    # --- Tab 4 (Baru): Klasterisasi Pelanggan ---
    with tab4:
        st.subheader("👥 Klasterisasi Pelanggan")
        st.markdown("---")

        st.markdown("""
        Klasterisasi pelanggan ini dilakukan berdasarkan dua fitur utama yang diekstraksi dari data ulasan Anda:
        1.  **Score (Rating Produk):** Ini adalah nilai rating numerik asli yang diberikan oleh pengguna untuk setiap produk.
        2.  **Sentiment Score (Skor Sentimen Ulasan):** Nilai ini diperoleh dari hasil analisis sentimen pada teks ulasan, menunjukkan seberapa positif atau negatif ulasan tersebut dalam skala numerik.

        Data awal yang digunakan untuk klasterisasi diambil dari kedua kolom ini. Penting untuk dicatat bahwa baris dengan nilai yang tidak lengkap atau hilang (NaN) di salah satu dari kedua kolom ini akan diabaikan untuk memastikan kualitas dan konsistensi analisis klaster.
        """)

        # Menggunakan df_sentiment dari session state untuk klasterisasi
        data_for_clustering_raw = st.session_state.df_sentiment if st.session_state.df_sentiment is not None else df
        
        # Pastikan kolom 'Text', 'Score', dan 'SentimentScore' ada
        if data_for_clustering_raw is not None and "Score" in data_for_clustering_raw.columns and "SentimentScore" in data_for_clustering_raw.columns and "Text" in data_for_clustering_raw.columns:
            # Memasukkan kolom 'Text' ke dalam data klasterisasi
            cluster_data = data_for_clustering_raw[["ProductId", "Score", "SentimentScore", "Text"]].dropna(subset=["Score", "SentimentScore", "Text"])
            st.markdown("**📋 Data untuk Klasterisasi (contoh 5 baris):**")
            st.dataframe(make_arrow_safe(cluster_data.head()))

            if len(cluster_data) < 5:
                st.warning("⚠️ Data terlalu sedikit untuk klasterisasi. Diperlukan minimal 5 baris data dengan 'Score', 'SentimentScore', dan 'Text' yang valid.")
            else:
                scaler = StandardScaler()
                # Scaling hanya pada kolom numerik yang digunakan untuk clustering
                scaled = scaler.fit_transform(cluster_data[["Score", "SentimentScore"]])
                
                st.markdown("### Metode Elbow untuk Menentukan Jumlah Klaster")
                inertia = []
                max_k = min(10, len(cluster_data) - 1)
                
                if max_k < 2:
                    st.warning("Data terlalu sedikit untuk menjalankan Elbow Method.")
                else:
                    for k in range(1, max_k + 1):
                        kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init='auto')
                        kmeans_temp.fit(scaled)
                        inertia.append(kmeans_temp.inertia_)

                    fig_elbow = go.Figure(data=go.Scatter(x=list(range(1, max_k + 1)), y=inertia, mode='lines+markers'))
                    fig_elbow.update_layout(title='Elbow Method untuk K-Means',
                                            xaxis_title='Jumlah Klaster (K)',
                                            yaxis_title='Inertia')
                    st.plotly_chart(fig_elbow)
                    st.info("Pilih jumlah klaster (K) di mana 'siku' tajam muncul, menunjukkan penurunan inersia yang signifikan melambat.")

                num_clusters = st.slider("Pilih Jumlah Klaster (K)", min_value=2, max_value=max_k, value=3)
                
                kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
                clusters = kmeans.fit_predict(scaled)
                cluster_data["Cluster"] = clusters

                fig = go.Figure()
                cmap = colormaps.get_cmap('viridis')
                for i in range(num_clusters):
                    points = cluster_data[cluster_data["Cluster"] == i]
                    norm_index = i / (num_clusters - 1) if num_clusters > 1 else 0.5
                    rgba_color = cmap(norm_index)
                    fig.add_trace(go.Scatter(
                        x=points["Score"],
                        y=points["SentimentScore"],
                        mode="markers",
                        name=f"Cluster {i}",
                        marker=dict(color=f'rgba({int(rgba_color[0]*255)},{int(rgba_color[1]*255)},{int(rgba_color[2]*255)},1)')
                    ))

                fig.update_layout(
                    title="Klaster Pelanggan Berdasarkan Rating & Sentimen",
                    xaxis_title="Score",
                    yaxis_title="Sentiment Score",
                    height=500
                )
                st.plotly_chart(fig)

                st.markdown("### 📊 Statistik per Klaster")
                cluster_summary = cluster_data.groupby("Cluster")[["Score", "SentimentScore"]].agg(["mean", "count"])
                st.dataframe(make_arrow_safe(cluster_summary))

                st.markdown("### 🧠 Interpretasi Klaster")
                for i, row in cluster_summary.iterrows():
                    score_mean = row[('Score', 'mean')]
                    sent_mean = row[('SentimentScore', 'mean')]
                    count = row[('Score', 'count')]

                    sentiment_label = "tinggi" if sent_mean >= 0.7 else "rendah"
                    rating_label = "tinggi" if score_mean >= 4 else ("rendah" if score_mean <= 2.5 else "sedang")

                    st.markdown(f"- **Cluster {i}** → Rata-rata rating **{score_mean:.2f}** ({rating_label}), rata-rata sentimen **{sent_mean:.2f}** ({sentiment_label}), total anggota: **{count}**.")
                
                # --- Bagian Baru: Tampilan Text Review per Klaster ---
                st.markdown("### 📝 Preview Ulasan Asli per Klaster")
                st.markdown("Pilih klaster-klaster yang ingin Anda lihat ulasan aslinya.")

                # Dapatkan daftar klaster yang tersedia
                available_clusters = sorted(cluster_data['Cluster'].unique().tolist())
                
                # Widget multiselect untuk memilih klaster
                selected_clusters = st.multiselect(
                    "Pilih Klaster (bisa lebih dari satu):",
                    options=available_clusters,
                    default=available_clusters if available_clusters else [] # Default pilih semua jika ada
                )

                if selected_clusters:
                    # Filter data berdasarkan klaster yang dipilih
                    filtered_reviews = cluster_data[cluster_data['Cluster'].isin(selected_clusters)]
                    
                    # Tampilkan tabel ulasan
                    st.dataframe(make_arrow_safe(filtered_reviews[['Cluster', 'ProductId', 'Score', 'SentimentScore', 'Text']]))
                else:
                    st.info("Pilih setidaknya satu klaster untuk menampilkan ulasan.")
                # --- Akhir Bagian Baru ---

        else:
            st.warning("Kolom 'Score', 'SentimentScore', atau 'Text' tidak ditemukan, atau analisis sentimen belum dijalankan. Harap jalankan analisis sentimen di tab 'Sentimen' terlebih dahulu.")

    # --- Tab 5: Kesimpulan Analisis ---
    with tab5:
        st.subheader("📝 Kesimpulan Analisis")
        st.markdown("---") 

        if st.session_state.df_sentiment is not None and 'Sentiment' in st.session_state.df_sentiment.columns:
            st.markdown("### 💬 Komposisi Sentimen")
            current_df = st.session_state.df_sentiment.copy()
            dist = current_df['Sentiment'].value_counts(normalize=True) * 100
            for label, pct in dist.items():
                st.markdown(f"   - **{label}**: {pct:.2f}% dari total ulasan")

            st.markdown("### 🧁 Diagram Komposisi Sentimen")
            fig_sent = go.Figure(data=[
                go.Pie(labels=dist.index, values=dist.values, hole=0.3, marker=dict(colors=['lightgreen', 'salmon']))
            ])
            fig_sent.update_layout(height=500, showlegend=True)
            st.plotly_chart(fig_sent, use_container_width=True)
        else:
            st.info("Analisis sentimen perlu dijalankan untuk menampilkan ringkasan komposisi sentimen.")

        if "ProductId" in df.columns and "Score" in df.columns:
            current_top_n = st.session_state.top_n_rekom
            current_min_reviews = st.session_state.min_reviews_rekom
            current_sort_option = st.session_state.sort_option_rekom
            
            rekom_df_for_conclusion = df.groupby("ProductId")["Score"].agg(AvgRating="mean", NumReviews="count")
            rekom_df_for_conclusion = rekom_df_for_conclusion[rekom_df_for_conclusion["NumReviews"] >= current_min_reviews].sort_values(by="AvgRating", ascending=False)
            rekom_df_for_conclusion = rekom_df_for_conclusion.sort_values(by=current_sort_option, ascending=False).head(current_top_n)

            if not rekom_df_for_conclusion.empty:
                st.markdown("### 🎯 Rekomendasi Teratas")
                top_item = rekom_df_for_conclusion['AvgRating'].idxmax()
                top_score = rekom_df_for_conclusion.loc[top_item, 'AvgRating']
                st.markdown(f"   - Produk dengan rating tertinggi (berdasarkan pengaturan di tab Rekomendasi): **{top_item}** dengan skor **{top_score:.2f}**")

                st.markdown("### 📊 Top Produk Rekomendasi")
                # Menggunakan warna dinamis di sini juga
                rekom_colors_conc = [get_rating_color(rating) for rating in rekom_df_for_conclusion['AvgRating']]
                
                # Membuat teks hover kustom untuk visualisasi rekomendasi di kesimpulan
                rekom_hover_texts_conc = [
                    f"Produk: {row.name}<br>"
                    f"Jumlah Ulasan: {row['NumReviews']}<br>"
                    f"Rata-rata Rating: {row['AvgRating']:.2f}"
                    for index, row in rekom_df_for_conclusion.iterrows()
                ]

                fig_rekom = go.Figure(go.Bar(
                    x=rekom_df_for_conclusion['AvgRating'],
                    y=rekom_df_for_conclusion.index,
                    orientation='h',
                    marker_color=rekom_colors_conc,
                    text=rekom_df_for_conclusion["AvgRating"].round(2),
                    textposition='auto',
                    hoverinfo='text',
                    hovertext=rekom_hover_texts_conc
                ))
                fig_rekom.update_layout(height=300, xaxis_title='Rating', yaxis_title='Product ID', title=f'Top {current_top_n} Produk Rekomendasi (Min. Review: {current_min_reviews})', xaxis_range=[0,5])
                st.plotly_chart(fig_rekom, use_container_width=True)
            else:
                st.info("Tidak ada produk yang memenuhi kriteria rekomendasi berdasarkan pengaturan saat ini di tab 'Rekomendasi'.")
        else:
            st.info("Kolom 'ProductId' atau 'Score' tidak ditemukan untuk menghitung rekomendasi di ringkasan.")


        cluster_data_for_conclusion = None
        data_for_conclusion_clustering = st.session_state.df_sentiment if st.session_state.df_sentiment is not None else df

        if data_for_conclusion_clustering is not None and "Score" in data_for_conclusion_clustering.columns and "SentimentScore" in data_for_conclusion_clustering.columns:
            temp_cluster_data = data_for_conclusion_clustering[["Score", "SentimentScore"]].dropna()
            if len(temp_cluster_data) >= 2:
                scaler_conc = StandardScaler()
                scaled_conc = scaler_conc.fit_transform(temp_cluster_data)
                num_clusters_conc = min(3, len(temp_cluster_data)-1) if len(temp_cluster_data) >= 3 else 2 
                if num_clusters_conc > 0:
                    kmeans_conc = KMeans(n_clusters=num_clusters_conc, random_state=42, n_init='auto')
                    clusters_conc = kmeans_conc.fit_predict(scaled_conc)
                    temp_cluster_data["Cluster"] = clusters_conc
                    cluster_data_for_conclusion = temp_cluster_data.copy()

        if cluster_data_for_conclusion is not None and 'Cluster' in cluster_data_for_conclusion.columns:
            st.markdown("### 👥 Klasterisasi Pelanggan")
            cluster_count = cluster_data_for_conclusion['Cluster'].value_counts().idxmax()
            count = cluster_data_for_conclusion['Cluster'].value_counts().max()
            st.markdown(f"   - Klaster pelanggan terbanyak: **Cluster {cluster_count}** dengan **{count}** anggota")
        else:
            st.info("Klasterisasi pelanggan perlu dijalankan di tab 'Klasterisasi' untuk menampilkan ringkasan.")

        st.markdown("---")
        st.subheader("💡 Aturan Metrik Evaluasi Model Sentimen")
        metrics = st.session_state.sentiment_metrics
        acc_val = metrics.get('accuracy', 'N/A')
        prec_val = metrics.get('precision', 'N/A')
        f1_val = metrics.get('f1', 'N/A')
        roc_auc_val = metrics.get('roc_auc', 'N/A')


        st.markdown("""
        Metrik evaluasi model sentimen yang digunakan umumnya memiliki rentang nilai **0 hingga 1**.
        """)
        
        st.markdown(f"**Akurasi (Accuracy):** **{acc_val:.2f}**" if isinstance(acc_val, float) else f"**Akurasi (Accuracy):** {acc_val}")
        st.markdown("""
            * **Apa itu?** Proporsi prediksi yang benar (baik positif maupun negatif) dari total prediksi.
            * **Aturan:** Semakin tinggi nilainya (mendekati 1), semakin baik model tersebut dalam memprediksi kelas secara keseluruhan.
        """)
        
        st.markdown(f"**Presisi (Precision):** **{prec_val:.2f}**" if isinstance(prec_val, float) else f"**Presisi (Precision):** {prec_val}")
        st.markdown("""
            * **Apa itu?** Proporsi prediksi positif yang sebenarnya benar-benar positif. Ini menjawab pertanyaan: "Dari semua ulasan yang saya prediksi positif, berapa banyak yang benar?"
            * **Aturan:** Semakin tinggi nilainya (mendekati 1), semakin sedikit *false positive* (model tidak salah mengklasifikasikan negatif sebagai positif).
        """)
        
        st.markdown(f"**F1-Score:** **{f1_val:.2f}**" if isinstance(f1_val, float) else f"**F1-Score:** {f1_val}")
        st.markdown("""
            * **Apa itu?** Rata-rata harmonik dari Presisi dan Recall. Ini adalah metrik yang baik ketika Anda membutuhkan keseimbangan antara presisi dan recall, terutama jika distribusi kelas tidak seimbang.
            * **Aturan:** Semakin tinggi nilainya (mendekati 1), semakin seimbang dan baik model dalam mengidentifikasi kelas positif.
        """)
        
        st.markdown(f"**ROC AUC (Receiver Operating Characteristic - Area Under Curve):** **{roc_auc_val:.2f}**" if isinstance(roc_auc_val, float) else f"**ROC AUC (Receiver Operating Characteristic - Area Under Curve):** {roc_auc_val}")
        st.markdown("""
            * **Apa itu?** Mengukur kemampuan model untuk membedakan antara kelas positif dan negatif. Nilai 0.5 berarti model tidak lebih baik dari tebakan acak, sedangkan 1.0 berarti model sempurna dalam membedakan.
            * **Aturan:** Semakin tinggi nilainya (mendekati 1), semakin baik kemampuan model dalam memisahkan kedua kelas.
                * **0.9 - 1.0**: Sangat baik
                * **0.8 - 0.9**: Baik
                * **0.7 - 0.8**: Cukup
                * **< 0.7**: Perlu peningkatan
        
        **Secara umum, untuk semua metrik ini, nilai yang lebih tinggi (mendekati 1) menunjukkan kinerja model yang lebih baik.**
        """)

        st.markdown("---")
        st.subheader("🧠 Kesimpulan") 
        kesimpulan = []
        
        # Penanganan kasus ketika df_sentiment belum ada
        if st.session_state.df_sentiment is not None and 'Sentiment' in st.session_state.df_sentiment.columns:
            current_df_for_conclusion = st.session_state.df_sentiment.copy() # Menggunakan df_sentiment jika sudah ada
        else:
            current_df_for_conclusion = df.copy() # Menggunakan df asli jika sentiment belum diproses

        if 'Sentiment' in current_df_for_conclusion.columns:
            total = current_df_for_conclusion.shape[0]
            dist = current_df_for_conclusion['Sentiment'].value_counts(normalize=True) * 100
            pos_pct = dist.get("POSITIVE", 0)
            neg_pct = dist.get("NEGATIVE", 0)
            kesimpulan.append(f"Analisis terhadap **{total:,}** ulasan pelanggan menunjukkan bahwa sekitar **{pos_pct:.1f}%** bernada positif dan **{neg_pct:.1f}%** negatif.")

            metrics = st.session_state.sentiment_metrics
            if metrics:
                acc = metrics.get('accuracy', 'N/A')
                prec = metrics.get('precision', 'N/A')
                f1 = metrics.get('f1', 'N/A')
                roc_auc = metrics.get('roc_auc', 'N/A')

                if isinstance(roc_auc, float):
                    kesimpulan.append(f"Model sentimen mencapai **Akurasi {acc:.2f}**, **Presisi {prec:.2f}**, **F1-Score {f1:.2f}**, dan nilai ROC AUC **{roc_auc:.2f}**.")
                    if roc_auc >= 0.9:
                        kesimpulan.append("Ini menunjukkan model memiliki kemampuan yang **sangat baik** dalam membedakan antara sentimen positif dan negatif.")
                    elif roc_auc >= 0.8:
                        kesimpulan.append("Ini menunjukkan model memiliki kemampuan yang **baik** dalam membedakan antara sentimen positif dan negatif.")
                    elif roc_auc >= 0.7:
                        kesimpulan.append("Ini menunjukkan model memiliki kemampuan yang **cukup baik** dalam membedakan sentimen.")
                    else:
                        kesimpulan.append("Namun, nilai ROC AUC menunjukkan bahwa kinerja model perlu **ditingkatkan** dalam membedakan sentimen.")
                else:
                    kesimpulan.append("Metrik evaluasi sentimen belum tersedia atau ROC AUC tidak dapat dihitung karena keterbatasan data.")
            else:
                kesimpulan.append("Metrik evaluasi sentimen belum tersedia.")
        else:
            kesimpulan.append("Analisis sentimen belum dijalankan. Untuk kesimpulan yang lebih lengkap, harap jalankan analisis sentimen.")


        if "ProductId" in df.columns and "Score" in df.columns:
            current_top_n_conc = st.session_state.top_n_rekom
            current_min_reviews_conc = st.session_state.min_reviews_rekom
            current_sort_option_conc = st.session_state.sort_option_rekom

            rekom_df_for_conclusion_text = df.groupby("ProductId")["Score"].agg(AvgRating="mean", NumReviews="count")
            rekom_df_for_conclusion_text = rekom_df_for_conclusion_text[rekom_df_for_conclusion_text["NumReviews"] >= current_min_reviews_conc].sort_values(by="AvgRating", ascending=False)
            rekom_df_for_conclusion_text = rekom_df_for_conclusion_text.sort_values(by=current_sort_option_conc, ascending=False).head(current_top_n_conc)

            if not rekom_df_for_conclusion_text.empty:
                top_item_conc = rekom_df_for_conclusion_text['AvgRating'].idxmax()
                top_score_conc = rekom_df_for_conclusion_text.loc[top_item_conc, 'AvgRating']
                kesimpulan.append(f"Produk dengan performa terbaik, berdasarkan pengaturan rekomendasi Anda (Top **{current_top_n_conc}**, Min. Review **{current_min_reviews_conc}**, diurutkan berdasarkan **{current_sort_option_conc}**), adalah **{top_item_conc}** dengan rata-rata rating **{top_score_conc:.2f}**.")
            else:
                kesimpulan.append("Tidak ada produk yang memenuhi kriteria rekomendasi Anda untuk disimpulkan.")
        else:
            kesimpulan.append("Informasi produk tidak lengkap untuk rekomendasi di kesimpulan.")

        cluster_data_for_conclusion = None
        data_for_conclusion_clustering = st.session_state.df_sentiment if st.session_state.df_sentiment is not None else df

        if data_for_conclusion_clustering is not None and "Score" in data_for_conclusion_clustering.columns and "SentimentScore" in data_for_conclusion_clustering.columns:
            temp_cluster_data = data_for_conclusion_clustering[["Score", "SentimentScore"]].dropna()
            if len(temp_cluster_data) >= 2:
                scaler_conc = StandardScaler()
                scaled_conc = scaler_conc.fit_transform(temp_cluster_data)
                num_clusters_conc = min(3, len(temp_cluster_data)-1) if len(temp_cluster_data) >= 3 else 2 
                if num_clusters_conc > 0:
                    kmeans_conc = KMeans(n_clusters=num_clusters_conc, random_state=42, n_init='auto')
                    clusters_conc = kmeans_conc.fit_predict(scaled_conc)
                    temp_cluster_data["Cluster"] = clusters_conc
                    cluster_data_for_conclusion = temp_cluster_data.copy()

        if cluster_data_for_conclusion is not None and 'Cluster' in cluster_data_for_conclusion.columns:
            st.markdown("### 👥 Klasterisasi Pelanggan")
            cluster_summary_conc = cluster_data_for_conclusion.groupby("Cluster")[["Score", "SentimentScore"]].agg(["mean", "count"])

            if not cluster_summary_conc.empty:
                largest_cluster_id = cluster_summary_conc[('Score', 'count')].idxmax()
                largest_cluster_count = cluster_summary_conc.loc[largest_cluster_id, ('Score', 'count')]
                largest_cluster_score_mean = cluster_summary_conc.loc[largest_cluster_id, ('Score', 'mean')]
                largest_cluster_sent_mean = cluster_summary_conc.loc[largest_cluster_id, ('SentimentScore', 'mean')]

                sentiment_label_lc = "tinggi" if largest_cluster_sent_mean >= 0.7 else ("rendah" if largest_cluster_sent_mean <= 0.3 else "sedang")
                rating_label_lc = "tinggi" if largest_cluster_score_mean >= 4 else ("rendah" if largest_cluster_score_mean <= 2.5 else "sedang")

                kesimpulan.append(f"Segmentasi pelanggan Anda menghasilkan beberapa klaster. Klaster terbesar adalah **Cluster {largest_cluster_id}** dengan **{largest_cluster_count}** anggota, yang rata-rata memiliki rating **{largest_cluster_score_mean:.2f}** ({rating_label_lc}) dan sentimen **{largest_cluster_sent_mean:.2f}** ({sentiment_label_lc}).")
            else:
                kesimpulan.append("Statistik klasterisasi tidak dapat dihitung untuk kesimpulan.")
        else:
            kesimpulan.append("Informasi klasterisasi pelanggan tidak tersedia di kesimpulan.")

        st.markdown("\n".join(kesimpulan))