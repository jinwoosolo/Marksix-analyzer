import streamlit as st
import pandas as pd
import plotly.express as px
from collections import Counter

st.set_page_config(layout="wide")
st.title("🎰 Mark Six Analyzer")

@st.cache_data(ttl=3600)
def load_data():
    df = pd.read_csv('marksix.csv')
    
    # 修日期格式
    df['date_parsed'] = pd.to_datetime(df['date'], format='%d/%m/%Y', errors='coerce')
    df = df.sort_values('date_parsed').reset_index(drop=True)
    
    # Debug
    st.sidebar.write(f"📅 Range: {df['date'].iloc[0]} → {df['date'].iloc[-1]}")
    st.sidebar.write(f"📊 {len(df)} total draws")
    
    return df




df = load_data()
st.write("### 🔍 Data Debug")
st.write(f"Date range: **{df['date'].min()} → {df['date'].max()}**")
st.write(f"Total draws: **{len(df):,d}**")
st.write(f"2026 draws: **{len(df[df['date'].str.contains('2026', na=False)]):,d}**")

st.success(f"✅ Loaded | Latest known: {df['date'].max()}")



st.success(f"✅ Loaded {len(df)} draws | Latest: {df['date'].max()}")

# === Metrics（安全版）===
col1, col2 = st.columns(2)
latest = df.iloc[-1]
col1.metric("Date", latest['date'])
nums = f"{latest['n1']}-{latest['n2']}-{latest['n3']}-{latest['n4']}-{latest['n5']}-{latest['n6']}"
col2.metric("Numbers", nums)

if 'div1_prize' in df.columns and pd.notna(latest.get('div1_prize')):
    col3, _ = st.columns(2)
    col3.metric("1st Prize", latest['div1_prize'])

# === 熱門號碼 ===
st.header("🔥 Hot Numbers (Recent)")
recent = df.tail(52)
all_nums = []
for _, row in recent.iterrows():
    nums = [row[f'n{i}'] for i in range(1, 7)]
    all_nums.extend(nums)

hot = Counter(all_nums).most_common(10)
fig = px.bar(x=[f"{x[0]}({x[1]})" for x in hot], 
             y=[x[1] for x in hot],
             color=[x[1] for x in hot],
             title="Most Frequent Numbers")
st.plotly_chart(fig, use_container_width=True)

# === 預測 ===
st.header("🤖 Next Draw Suggestion")
pred = [hot[i][0] for i in range(3)]
st.balloons()
st.success(f"**Play these: {' | '.join(map(str, sorted(pred)))}**")

# === 趨勢 ===
st.header("📈 Trends")
df['total'] = df[[f'n{i}' for i in range(1,7)]].sum(axis=1)
fig_trend = px.line(df.tail(50), x='date', y='total', 
                   title="Number Sum (Recent)")
st.plotly_chart(fig_trend)

st.markdown("---")
st.caption("🚀 Auto-updating Mark Six analyzer")
