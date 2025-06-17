import streamlit as st
from transformers import pipeline
import plotly.graph_objects as go

# Load pipeline
sent_pipeline = pipeline("sentiment-analysis")

st.title("Amazon Food Review Sentiment Analysis")

user_input = st.text_area("Enter a sentence for analysis:")

if st.button("Analysis Sentiment"):
    if user_input.strip() == "":
        st.warning("Filed Required.")
    else:
        with st.spinner("Analysis..."):
            result = sent_pipeline(user_input)
            label = result[0]['label']
            score = result[0]['score']

            st.subheader("Result Analysis:")
            st.write(f"**Label Sentimen:** {label}")
            st.write(f"**Score Belief:** {score:.4f}")

            # Visualisasi dengan Plotly
            fig = go.Figure(go.Bar(
                x=[score],
                y=[label],
                orientation='h',
                marker_color='skyblue',
                text=[f"{score:.2f}"],
                textposition='auto'
            ))

            fig.update_layout(
                xaxis=dict(range=[0, 1]),
                title="Sentiment Score Visualization",
                height=300
            )

            st.plotly_chart(fig)
