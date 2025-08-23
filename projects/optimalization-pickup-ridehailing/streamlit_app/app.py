import streamlit as st
import pandas as pd
from utils.db import engine

st.title("CONNECTION TEST DAN DATA QUERY")

try:
    with engine.connect() as conn:
        # server time test
        time_df = pd.read_sql("SELECT NOW()", conn)
        st.success(f"✅ Connection Success! Server Time: {time_df.iloc[0, 0]}")

        # query to supabase table
        query = "SELECT * FROM public.jakarta_ride_trips LIMIT 10"
        df = pd.read_sql(query, conn)

        # Visualize Dataframe
        st.subheader("📊 Example data from jakarta_ride_trips")
        st.dataframe(df)

except Exception as e:
    st.error(f"❌ failure connection or query: {e}")
