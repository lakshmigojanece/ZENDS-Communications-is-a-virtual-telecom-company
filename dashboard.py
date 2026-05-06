import streamlit as st
import pandas as pd
import os

st.title("📊 Support Analytics Dashboard")

# Absolute path fix
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data", "ZenDS_Communications_20k_queries_final (1).csv")

df = pd.read_csv(data_path)

st.subheader("Intent Distribution")
st.bar_chart(df["intent"].value_counts())

st.subheader("Sentiment Distribution")
st.bar_chart(df["sentiment"].value_counts())
