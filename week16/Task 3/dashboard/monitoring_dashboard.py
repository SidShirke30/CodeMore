import streamlit as st
import pandas as pd
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data" / "metrics.csv"

st.set_page_config(page_title="AI Model Monitoring", layout="wide")
st.title("AI Model Performance Dashboard")

df = pd.read_csv(DATA)

c1, c2, c3 = st.columns(3)
c1.metric("Latest Accuracy", f"{df['accuracy'].iloc[-1]:.2%}")
c2.metric("Latest Latency", f"{df['latency_ms'].iloc[-1]:.0f} ms")
c3.metric("CPU Utilization", f"{df['cpu_percent'].iloc[-1]:.1f}%")

st.subheader("Accuracy Over Time")
st.line_chart(df.set_index("timestamp")["accuracy"])

st.subheader("Latency Over Time")
st.line_chart(df.set_index("timestamp")["latency_ms"])

st.subheader("Resource Utilization")
st.line_chart(df.set_index("timestamp")[["cpu_percent", "memory_percent"]])

st.subheader("Collected Metrics")
st.dataframe(df, use_container_width=True)
