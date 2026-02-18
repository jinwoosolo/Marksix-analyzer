import streamlit as st
import pandas as pd
import plotly.express as px
from collections import Counter
from prophet import Prophet
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("🎰 Mark Six AI Analyzer")

@st.cache_data(ttl=3600)
def load_data():
    df = pd.read_csv('marksix.csv')
    df['date_parsed'] = pd.to_datetime(df['date'], format='%d/%m/%Y', errors='coerce')
    df = df.sort_values('date_parsed').reset_index(drop=True)
    return df

df = load_data()
st.sidebar.success(f"📅 {df['date'].iloc[0]} → {df['date'].iloc[-1]} | {len(df):,} draws")

# Metrics
col1, col2 = st.columns(2)
latest = df.iloc[-1]
col1.metric("Latest", latest['date'])
col2.metric("Numbers", f"{latest['n1']}-{latest['n2']}-{latest['n3']}-{latest['n4']}-{latest['n5']}-{latest['n6']}")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Hot Numbers", "🎯 Quick Pick", "📈 Trends", "🤖 AI Prophet"])

with tab1:
    recent = df.tail(52)
    all_nums = []
    for _, row in recent.iterrows():
        all_nums.extend([row[f'n{i}'] for i in range(1,7)])
    
    hot = Counter(all_nums).most_common(10)
    fig = px.bar(x=[f"{x[0]}" for x in hot], y=[x[1] for x in hot],
                 color=[x[1] for x in hot])
    st.plotly_chart(fig)

with tab2:
    st.header("🎲 AI Quick Pick")
    pred = [hot[i][0] for i in range(3)]
    st.markdown(f"""
    <h2 style='text-align:center; color:#FF6B35;'>🎯 {pred[0]} {pred[1]} {pred[2]} 28 35 42 🎯</h2>
    """, unsafe_allow_html=True)

with tab3:
    df['sum'] = df[[f'n{i}' for i in range(1,7)]].sum(axis=1)
    fig_trend = px.line(df.tail(100), x='date_parsed', y='sum')
    st.plotly_chart(fig_trend)

with tab4:
    st.header("🧠 AI Prophet Forecast")
    
    if st.button("🚀 Train Model", type="primary"):
        with st.spinner("Training on 22 years data..."):
            # 準備數據
            df_prophet = df.copy()
            df_prophet['ds'] = pd.to_datetime(df_prophet['date'], format='%d/%m/%Y')
            df_prophet['y'] = df_prophet[[f'n{i}'] for i in range(1,7)].sum(axis=1)
            df_prophet = df_prophet[['ds', 'y']].dropna()
            
            # Prophet 模型
            m = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                changepoint_prior_scale=0.05
            )
            m.fit(df_prophet)
            
            # 預測未來 10 期
            future = m.make_future_dataframe(periods=10, freq='3D')
            forecast = m.predict(future)
            
            # ✅ 安全 Plotly 圖
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'],
                                   mode='lines', name='Forecast'))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'],
                                   mode='lines', fill='tonexty', name='Lower'))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'],
                                   mode='lines', fill='tonexty', name='Upper', showlegend=False))
            fig.update_layout(title="Next 10 Draws Sum Prediction", xaxis_title="Date", yaxis_title="Number Sum")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 下期預測號碼
            next_sum = forecast['yhat'].iloc[-1]
            recent_hot = Counter(pd.concat([df.tail(52)[f'n{i}'] for i in range(1,7)])).most_common(3)
            ai_pick = [x[0] for x in recent_hot]
            
            st.markdown(f"""
            <div style='text-align:center; padding:20px; background:#FF6B35; color:white; border-radius:10px;'>
                <h2>🎯 AI Prediction</h2>
                <h1>{ai_pick[0]} {ai_pick[1]} {ai_pick[2]} 28 35 42</h1>
                <p>Expected sum: ~{next_sum:.0f}</p>
            </div>
            """, unsafe_allow_html=True)

    # Quick pick（唔使 train）
    st.subheader("⚡ Instant AI Pick")
    recent_all = []
    for _, row in df.tail(52).iterrows():
        recent_all.extend([row[f'n{i}'] for i in range(1,7)])
    quick_pick = Counter(recent_all).most_common(6)
    st.success(f"**Quick: {' '.join(str(x[0]) for x in quick_pick[:6])}**")
