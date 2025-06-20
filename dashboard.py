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

# ------------------------------
# Konfigurasi Streamlit
# ------------------------------
st.set_page_config(layout="wide")
st.set_option('server.maxUploadSize', 400)
st.title("📊 Dashboard Analisis Perilaku Pelanggan & Rekomendasi Produk")

st.sidebar.header("🎯 Tujuan Analisis")
st.sidebar.markdown("""
- Menemukan pola perilaku pelanggan
- Melakukan analisis sentimen ulasan
- Menyediakan sistem rekomendasi sederhana
- Klasterisasi pelanggan
""")

# ------------------------------
# Helper: Konversi DataFrame agar aman untuk Arrow (st.dataframe)
def make_arrow_safe(df):
    df = df.copy()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str)
    return df

# ------------------------------
# Inisialisasi session_state untuk df_sentiment
if 'df_sentiment' not in st.session_state:
    st.session_state.df_sentiment = None

# ------------------------------
# Upload Data
# ------------------------------
st.sidebar.header("📥 Upload Dataset CSV")
uploaded_file = st.sidebar.file_uploader("Upload file CSV", type=["csv"])

if uploaded_file:
    if st.session_state.df_sentiment is not None:
        df = st.session_state.df_sentiment.copy()
    else:
        row_limit = st.sidebar.slider("🔢 Jumlah data yang ingin dimuat (max)", min_value=100, max_value=500000, value=1000, step=500)
        df = pd.read_csv(uploaded_file).head(row_limit)
        df = df.convert_dtypes()
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str)
        st.success(f"✅ Data berhasil dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")

    # ------------------------------
    # Tabs Utama
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Eksplorasi Data (EDA)", "💬 Sentimen", "🎯 Rekomendasi", "👥 Klasterisasi", "📝 Kesimpulan"])

    with tab1:
        st.subheader("🔍 Eksplorasi Data (EDA)")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Distribusi Rating**")
            if "Score" in df.columns:
                st.markdown("Diagram ini menunjukkan sebaran jumlah rating produk dari pengguna.")
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

        st.markdown("### Statistik Deskriptif (Numerik)")
        st.caption("⚠️ Statistik ini hanya dihitung dari subset data yang dimuat (maksimum " + str(df.shape[0]) + " baris).")
        numeric_desc = df.select_dtypes(include=[np.number]).describe().T
        st.dataframe(make_arrow_safe(numeric_desc))

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

    with tab2:
        st.markdown("""
---
## 🔍 Pemodelan Sentimen
Silakan jalankan analisis sentimen ulasan pelanggan untuk mendapatkan insight lebih lanjut.
""")
        st.subheader("💬 Analisis Sentimen dengan Transformers")
        sent_pipeline = pipeline("sentiment-analysis")

        if st.button("🔍 Jalankan Analisis Sentimen"):
            with st.spinner("Sedang memproses..."):
                sentiment_labels = []
                sentiment_scores = []
                import time
                start_time = time.time()
                progress = st.progress(0, text="🔄 Memproses sentimen...")
                texts = df['Text'].astype(str).tolist()
                batch_size = 32
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i+batch_size]
                    batch = [text[:512] for text in batch] # Truncate text for model input
                    results = sent_pipeline(batch)
                    for res in results:
                        sentiment_labels.append(res['label'])
                        sentiment_scores.append(res['score'])
                    percent_complete = min((i + batch_size) / len(texts), 1.0)
                    est_total = (time.time() - start_time) / percent_complete
                    est_remaining = est_total - (time.time() - start_time)
                    progress.progress(percent_complete, text=f"⏳ {est_remaining:.1f} detik tersisa") # Updated text for remaining time
                elapsed = time.time() - start_time
                st.success(f"⏱️ Waktu proses: {elapsed:.2f} detik")
                df['Sentiment'] = sentiment_labels
                df['SentimentScore'] = sentiment_scores
            st.session_state.df_sentiment = df.copy() # Simpan hasil sentimen ke session state

            st.success("✅ Analisis sentimen selesai!")

            # Evaluasi Model Sentimen
            from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, f1_score, roc_auc_score

            if 'Score' in df.columns and df['Sentiment'].isin(['POSITIVE', 'NEGATIVE']).all():
                df['SentimentTrue'] = df['Score'].apply(lambda x: 'POSITIVE' if x >= 3 else 'NEGATIVE') # Asumsi score >= 3 adalah positif
                y_true = df['SentimentTrue']
                y_pred = df['Sentiment']
                
                # Check for empty y_true or y_pred
                if len(y_true) > 0 and len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1: # Ensure at least two classes for report/AUC
                    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

                    accuracy = accuracy_score(y_true, y_pred)
                    precision = precision_score(y_true, y_pred, pos_label='POSITIVE', zero_division=0)
                    f1 = f1_score(y_true, y_pred, pos_label='POSITIVE', zero_division=0)
                    try:
                        roc_auc = roc_auc_score(y_true.map({'NEGATIVE': 0, 'POSITIVE': 1}), y_pred.map({'NEGATIVE': 0, 'POSITIVE': 1}))
                    except ValueError: # Handle case where only one class is present in true/pred
                        roc_auc = 'N/A (single class)'

                    st.subheader("📊 Evaluasi Model Sentimen")
                    st.markdown("**📋 Metrik Tambahan:**")
                    st.write({
                        'Accuracy': f"{accuracy:.2f}",
                        'Precision': f"{precision:.2f}",
                        'F1-Score': f"{f1:.2f}",
                        'ROC AUC': f"{roc_auc}"
                    })

                    st.dataframe(pd.DataFrame(report).transpose())

                    st.subheader("📌 Confusion Matrix")
                    cm = confusion_matrix(df['SentimentTrue'], df['Sentiment'], labels=['NEGATIVE', 'POSITIVE'])
                    fig, ax = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['NEGATIVE', 'POSITIVE'], yticklabels=['NEGATIVE', 'POSITIVE'])
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    st.pyplot(fig)
                else:
                    st.warning("Tidak cukup variasi label sentimen atau rating untuk menghitung metrik evaluasi model.")

        # Tampilkan hasil sentimen jika sudah ada di session state
        if st.session_state.df_sentiment is not None and 'Sentiment' in st.session_state.df_sentiment.columns:
            df = st.session_state.df_sentiment.copy() # Gunakan df dari session state yang sudah punya kolom sentimen
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

            # --- Visualisasi Tren Sentimen (Time Series) ---
            st.subheader("📈 Tren Sentimen dari Tahun ke Tahun")
            st.markdown("Visualisasi ini menunjukkan bagaimana proporsi sentimen positif dan negatif berubah sepanjang waktu.")
            
            # Cek apakah kolom 'Time' ada di DataFrame
            if 'Time' in df.columns:
                try:
                    # Konversi kolom 'Time' ke datetime (asumsi Unix timestamp atau format date string)
                    # Jika 'Time' adalah Unix timestamp, perlu dikalikan 1000 untuk milidetik
                    # atau langsung digunakan jika sudah dalam format detik
                    df_trend = df.copy()
                    df_trend['Datetime'] = pd.to_datetime(df_trend['Time'], unit='s', errors='coerce')
                    
                    # Hapus baris dengan nilai datetime yang tidak valid setelah konversi
                    df_trend.dropna(subset=['Datetime'], inplace=True)

                    if not df_trend.empty:
                        df_trend['Year'] = df_trend['Datetime'].dt.year
                        
                        # Hitung proporsi sentimen per tahun
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

    with tab3:
        st.subheader("🎯 Rekomendasi Produk Sederhana")
        if "ProductId" in df.columns and "Score" in df.columns:
            top_n = st.slider("Berapa produk tertinggi yang ingin ditampilkan?", 1, 20, 5)
            min_reviews = st.slider("Minimum jumlah review untuk direkomendasikan", 1, 100, 10)
            
            rekom_df = df.groupby("ProductId")["Score"].agg(AvgRating="mean", NumReviews="count")
            rekom_df = rekom_df[rekom_df["NumReviews"] >= min_reviews].sort_values(by="AvgRating", ascending=False) # Default sort by AvgRating
            
            sort_option = st.radio("Urutkan berdasarkan:", ["AvgRating", "NumReviews"], horizontal=True, key="sort_rekom")
            rekom_df = rekom_df.sort_values(by=sort_option, ascending=False).head(top_n)
            
            st.dataframe(make_arrow_safe(rekom_df.reset_index()))

            st.subheader("📌 Visualisasi Produk Terbaik Berdasarkan Rating")
            if not rekom_df.empty:
                fig = go.Figure(go.Bar(
                    x=rekom_df["AvgRating"],
                    y=rekom_df.index,
                    orientation='h',
                    marker_color='teal',
                    text=rekom_df["AvgRating"].round(2),
                    textposition='auto'
                ))
                fig.update_layout(title="Top Produk Rekomendasi", xaxis_title="Rating Rata-rata")
                st.markdown("Visualisasi ini membantu memahami produk dengan rating tertinggi berdasarkan nilai rata-rata.")
                st.plotly_chart(fig)
            else:
                st.info("Tidak ada produk yang memenuhi kriteria rekomendasi.")

            # Insight tambahan untuk rekomendasi (hanya di tab ini)
            st.subheader("📌 Insight Rekomendasi Produk")
            
            # Pastikan rekom_df ada dan tidak kosong untuk filter selanjutnya
            if not rekom_df.empty:
                good = rekom_df[rekom_df['AvgRating'] >= 4.5]
                improve = rekom_df[(rekom_df['AvgRating'] >= 3.0) & (rekom_df['AvgRating'] < 4.0)]
                drop = rekom_df[rekom_df['AvgRating'] < 3.0]
            else:
                # Jika rekom_df kosong, buat DataFrame kosong untuk setiap kategori agar tidak error
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


    with tab4:
        st.subheader("👥 Klasterisasi Pelanggan")
        # Pastikan kolom Sentimen dan Score ada untuk klasterisasi
        if "Score" in df.columns and "SentimentScore" in df.columns and st.session_state.df_sentiment is not None:
            cluster_data = df[["Score", "SentimentScore"]].dropna()
            st.markdown("**📋 Data untuk Klasterisasi (contoh 5 baris):**")
            st.dataframe(make_arrow_safe(cluster_data.head()))

            if len(cluster_data) < 5:
                st.warning("⚠️ Data terlalu sedikit untuk klasterisasi. Diperlukan minimal 5 baris data dengan 'Score' dan 'SentimentScore' yang valid.")
            else:
                scaler = StandardScaler()
                scaled = scaler.fit_transform(cluster_data)
                
                # Menggunakan Elbow Method untuk menentukan jumlah cluster yang optimal
                st.markdown("### Metode Elbow untuk Menentukan Jumlah Klaster")
                inertia = []
                max_k = min(10, len(cluster_data) - 1) # Max k limited by data size
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
                # Use a color scale for more than 3 clusters
                colors = plt.cm.get_cmap('viridis', num_clusters)
                for i in range(num_clusters):
                    points = cluster_data[cluster_data["Cluster"] == i]
                    fig.add_trace(go.Scatter(
                        x=points["Score"],
                        y=points["SentimentScore"],
                        mode="markers",
                        name=f"Cluster {i}",
                        marker=dict(color=f'rgba({int(colors(i)[0]*255)},{int(colors(i)[1]*255)},{int(colors(i)[2]*255)},1)') # Convert to rgba
                    ))

                fig.update_layout(
                    title="Klaster Pelanggan Berdasarkan Rating & Sentimen",
                    xaxis_title="Score",
                    yaxis_title="Sentiment Score",
                    height=500
                )
                st.plotly_chart(fig)

                # Statistik ringkasan per klaster
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
        else:
            st.warning("Kolom 'Score' atau 'SentimentScore' tidak ditemukan, atau analisis sentimen belum dijalankan. Harap jalankan analisis sentimen di tab 'Sentimen' terlebih dahulu.")

    with tab5: # Tab baru untuk kesimpulan
        st.subheader("📝 Kesimpulan Analisis")
        st.markdown("Berikut adalah ringkasan dari analisis yang telah dilakukan:")

        # Pastikan data sentimen sudah ada sebelum menampilkan ringkasan sentimen
        if st.session_state.df_sentiment is not None and 'Sentiment' in st.session_state.df_sentiment.columns:
            st.markdown("### 💬 Komposisi Sentimen")
            # Menggunakan df yang sudah diperbarui dengan sentimen dari session state
            current_df = st.session_state.df_sentiment.copy()
            dist = current_df['Sentiment'].value_counts(normalize=True) * 100
            for label, pct in dist.items():
                st.markdown(f"   - **{label}**: {pct:.2f}% dari total ulasan")

            st.markdown("### 🧁 Diagram Komposisi Sentimen")
            fig_sent = go.Figure(data=[
                go.Pie(labels=dist.index, values=dist.values, hole=0.3, marker=dict(colors=['lightgreen', 'salmon']))
            ])
            fig_sent.update_layout(height=300, showlegend=True)
            st.plotly_chart(fig_sent, use_container_width=True)
        else:
            st.info("Analisis sentimen perlu dijalankan untuk menampilkan ringkasan komposisi sentimen.")

        # Rekomendasi Teratas
        # Pastikan rekom_df didefinisikan di tab3 sebelum diakses di sini
        # Mengambil rekom_df langsung dari hasil perhitungan di tab3
        if "ProductId" in df.columns and "Score" in df.columns:
            # Re-calculate rekom_df if not already available from tab3 (e.g., if user directly jumps to Conclusion)
            rekom_df = df.groupby("ProductId")["Score"].agg(AvgRating="mean", NumReviews="count")
            min_reviews_for_summary = 10 # Using a default minimum review for summary
            rekom_df = rekom_df[rekom_df["NumReviews"] >= min_reviews_for_summary].sort_values(by="AvgRating", ascending=False)
            
            if not rekom_df.empty:
                st.markdown("### 🎯 Rekomendasi Teratas")
                top_item = rekom_df['AvgRating'].idxmax()
                top_score = rekom_df.loc[top_item, 'AvgRating']
                st.markdown(f"   - Produk dengan rating tertinggi: **{top_item}** dengan skor **{top_score:.2f}**")

                st.markdown("### 📊 Top Produk Rekomendasi")
                fig_rekom = go.Figure(go.Bar(
                    x=rekom_df.head(5)['AvgRating'],
                    y=rekom_df.head(5).index,
                    orientation='h',
                    marker_color='orange'
                ))
                fig_rekom.update_layout(height=300, xaxis_title='Rating', yaxis_title='Product ID')
                st.plotly_chart(fig_rekom, use_container_width=True)
            else:
                st.info("Rekomendasi produk perlu dihasilkan di tab 'Rekomendasi' untuk menampilkan ringkasan.")
        else:
            st.info("Kolom 'ProductId' atau 'Score' tidak ditemukan untuk menghitung rekomendasi di ringkasan.")


        # Klasterisasi Pelanggan
        # Pastikan cluster_data didefinisikan di tab4 sebelum diakses di sini
        if 'cluster_data' in locals() and 'Cluster' in cluster_data.columns:
            st.markdown("### 👥 Klasterisasi Pelanggan")
            cluster_count = cluster_data['Cluster'].value_counts().idxmax()
            count = cluster_data['Cluster'].value_counts().max()
            st.markdown(f"   - Klaster pelanggan terbanyak: **Cluster {cluster_count}** dengan **{count}** anggota")
        else:
            st.info("Klasterisasi pelanggan perlu dijalankan di tab 'Klasterisasi' untuk menampilkan ringkasan.")

        st.markdown("### 🧠 Kesimpulan Otomatis")
        kesimpulan = []
        if st.session_state.df_sentiment is not None and 'Sentiment' in st.session_state.df_sentiment.columns:
            current_df = st.session_state.df_sentiment.copy()
            total = current_df.shape[0]
            dist = current_df['Sentiment'].value_counts(normalize=True) * 100
            pos_pct = dist.get("POSITIVE", 0)
            neg_pct = dist.get("NEGATIVE", 0)
            kesimpulan.append(f"Analisis terhadap **{total:,}** ulasan pelanggan menunjukkan bahwa sekitar **{pos_pct:.1f}%** bernada positif dan **{neg_pct:.1f}%** negatif.")

        # Recalculate rekom_df for conclusion if needed, to ensure it's available
        if "ProductId" in df.columns and "Score" in df.columns:
            rekom_df_for_conclusion = df.groupby("ProductId")["Score"].agg(AvgRating="mean", NumReviews="count")
            rekom_df_for_conclusion = rekom_df_for_conclusion[rekom_df_for_conclusion["NumReviews"] >= min_reviews_for_summary].sort_values(by="AvgRating", ascending=False)
            if not rekom_df_for_conclusion.empty:
                top_item_conc = rekom_df_for_conclusion['AvgRating'].idxmax()
                top_score_conc = rekom_df_for_conclusion.loc[top_item_conc, 'AvgRating']
                kesimpulan.append(f"Produk dengan performa terbaik adalah **{top_item_conc}** dengan rata-rata rating **{top_score_conc:.2f}**.")
            else:
                kesimpulan.append("Tidak ada produk yang memenuhi kriteria rekomendasi untuk kesimpulan.")
        else:
            kesimpulan.append("Informasi produk tidak lengkap untuk rekomendasi di kesimpulan.")


        if 'cluster_data' in locals() and 'Cluster' in cluster_data.columns:
            cluster_count = cluster_data['Cluster'].value_counts().idxmax()
            count = cluster_data['Cluster'].value_counts().max()
            kesimpulan.append(f"Segmentasi pelanggan menunjukkan bahwa **Cluster {cluster_count}** merupakan yang terbesar dengan **{count}** anggota.")
        else:
            kesimpulan.append("Informasi klasterisasi pelanggan tidak tersedia di kesimpulan.")

        st.markdown("\n".join(kesimpulan))