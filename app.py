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
    st.header("🧠 Prophet AI Forecast")
    if st.button("🚀 Train Model"):
        df_prophet = df.copy()
        df_prophet = df_prophet.rename(columns={'date_parsed': 'ds', 'sum': 'y'}).head(2000)
        m = Prophet(weekly_seasonality=True)
        m.fit(df_prophet[['ds', 'y']])
        future = m.make_future_dataframe(periods=10)
        forecast = m.predict(future)
        fig = plot_plotly(m, forecast)
        st.plotly_chart(fig)

st.caption("🤖 Auto-updating | 22 years data")
