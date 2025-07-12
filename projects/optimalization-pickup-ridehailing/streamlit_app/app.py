import streamlit as st
import pandas as pd
from utils.db import engine

st.title("Tes Koneksi Supabase")

try:
    with engine.connect() as conn:
        df = pd.read_sql("SELECT NOW()", conn)
        st.success(f"Koneksi berhasil! Waktu server: {df.iloc[0,0]}")
except Exception as e:
    st.error(f"Gagal koneksi: {e}")
