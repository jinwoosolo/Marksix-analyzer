import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from collections import Counter
import plotly.graph_objects as go

st.set_page_config(page_title="Mark Six Analyzer", layout="wide")

@st.cache_data(ttl=3600)  # Cache 1小時
def load_data():
    return pd.read_csv('marksix_49ball_full_2002_2026.csv')

df = load_data()
st.title("🎰 Mark Six Analyzer - Auto Updated")
st.caption(f"📊 {len(df):,} draws analyzed | Last updated: {df['date'].max()}")

# === 最新開彩 ===
col1, col2, col3 = st.columns(3)
latest = df.tail(1).iloc[0]
col1.metric("Latest Draw", latest['date'])
col2.metric("Numbers", f"{latest['n1']}-{latest['n2']}-{latest['n3']}-{latest['n4']}-{latest['n5']}-{latest['n6']}+{latest['extra']}")
if 'div1_prize' in df.columns:
    col3.metric("1st Prize", latest.get('div1_prize', 'N/A'))

# === 熱門號碼 ===
st.header("🔥 Hot Numbers (Last 52 Draws)")
recent_df = df.tail(52)
all_nums = []
for _, row in recent_df.iterrows():
    all_nums.extend([row[f'n{i}'] for i in range(1,7)] + [row['extra']])

hot = Counter(all_nums).most_common(10)
cold = Counter(all_nums).most_common()[-10:][::-1]

col1, col2 = st.columns(2)
with col1:
    st.subheader("Hot (Most Frequent)")
    fig_hot = px.bar(x=[x[0] for x in hot], y=[x[1] for x in hot], 
                     title="Hot Numbers", color=[x[1] for x in hot])
    st.plotly_chart(fig_hot, use_container_width=True)

with col2:
    st.subheader("Cold (Least Frequent)")
    fig_cold = px.bar(x=[x[0] for x in cold], y=[x[1] for x in cold],
                      title="Cold Numbers", color=[x[1] for x in cold])
    st.plotly_chart(fig_cold, use_container_width=True)

# === 預測 ===
st.header("🤖 Next Draw Prediction")
st.info("🔥 Hot + ❄️ Cold + 📈 Trend Analysis")

prediction = [hot[0][0], hot[1][0], cold[0][0], cold[1][0], hot[2][0], cold[2][0]]
st.success(f"**Recommended: {' '.join(map(str, sorted(prediction)))}**")

# === 統計 ===
tab1, tab2 = st.tabs(["📈 Trends", "🎯 Full Stats"])

with tab1:
    df['sum'] = df[[f'n{i}' for i in range(1,7)]].sum(axis=1)
    fig_trend = px.line(df.tail(100), x='date', y='sum', 
                       title="Number Sum Trend (Last 100 Draws)")
    st.plotly_chart(fig_trend)

with tab2:
    st.subheader("All Time Frequency")
    freq_cols = [f'n{i}' for i in range(1,7)] + ['extra']
    all_freq = pd.concat([df[col] for col in freq_cols])
    fig_freq = px.histogram(all_freq.value_counts().sort_index(), 
                           title="Number Frequency (All Time)")
    st.plotly_chart(fig_freq)

st.markdown("---")
st.caption("🤖 Auto-updated via GitHub Actions | Data from HKJC via lottery.hk")
