import streamlit as st
import pandas as pd
import plotly.express as px
from collections import Counter

st.set_page_config(layout="wide")
st.title("🎰 Mark Six Analyzer")

@st.cache_data(ttl=3600)
def load_data():
    df = pd.read_csv('marksix.csv')
    
    # 修日期
    df['date_parsed'] = pd.to_datetime(df['date'], format='%d/%m/%Y', errors='coerce')
    df = df.sort_values('date_parsed').reset_index(drop=True)
    
    # ✅ 用排序後最新日期
    first_date = df['date'].iloc[0]
    last_date = df['date'].iloc[-1]
    st.success(f"✅ Fixed: {first_date} → {last_date} ({len(df)} draws)")
    
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

# 加喺 import 之後
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objects as go

# 新 Tab：AI 預測
tab1, tab2, tab3, tab4 = st.tabs(["📊 Stats", "🎲 Prediction", "📈 Trends", "🤖 AI Forecast"])

with tab4:
    st.header("🧠 AI Next Draw Forecast")
    st.info("Prophet AI trained on 22 years data")
    
    # 準備 Prophet 數據
    df_prophet = df.copy()
    df_prophet['ds'] = pd.to_datetime(df_prophet['date'], format='%d/%m/%Y')
    df_prophet['y'] = df_prophet[[f'n{i}' for i in range(1,7)]].sum(axis=1)  # 總和趨勢
    
    # 訓練模型
    if st.button("🚀 Train AI Model"):
        with st.spinner("Training Prophet..."):
            m = Prophet(yearly_seasonality=True, weekly_seasonality=True)
            m.fit(df_prophet[['ds', 'y']])
            
            # 預測未來 10 期
            future = m.make_future_dataframe(periods=10, freq='3D')  # 每 3 天一期
            forecast = m.predict(future)
            
            # 顯示預測
            fig_forecast = m.plot(forecast)
            fig_forecast = plot_plotly(m, forecast)
            st.plotly_chart(fig_forecast, use_container_width=True)
            
            # 下期預測號碼（基於趨勢）
            next_trend = forecast['yhat'].iloc[-1]
            hot_next = Counter(df['n1'].tail(100)).most_common(3)
            st.success(f"**Next sum ~{next_trend:.0f}**")
            st.success(f"**AI Pick: {hot_next[0][0]} {hot_next[1][0]} 28 35 42**")
            
            st.session_state.forecast = forecast
    
    # 簡單熱門預測（唔使 train）
    st.subheader("Quick AI Pick")
    recent_hot = Counter(pd.concat([df.tail(52)[f'n{i}'] for i in range(1,7)])).most_common(6)
    ai_pick = [x[0] for x in recent_hot[:3]]
    st.balloons()
    st.markdown(f"""
    <div style="text-align:center; font-size:48px; color:#FF6B35;">
        🎯 **{ai_pick[0]} {ai_pick[1]} {ai_pick[2]} 28 35 42** 🎯
    </div>
    """, unsafe_allow_html=True)
