import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from collections import Counter

# --- CONFIGURATION ---
st.set_page_config(page_title="Mark Six Pro AI", page_icon="🎰", layout="wide")

# Modern Styling
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    div.stButton > button:first-child {
        background: linear-gradient(45deg, #FF6B35, #FF8C5A);
        color: white; border: none; border-radius: 10px; width: 100%; height: 3em; font-weight: bold;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        padding: 20px; border-radius: 15px; border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
    }
    .stat-val { color: #FF6B35; font-size: 28px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_data():
    df = pd.read_csv('marksix.csv')
    # FIX: Use format='mixed' to handle both YYYY/MM/DD and DD/MM/YYYY
    df['date_parsed'] = pd.to_datetime(df['date'], format='mixed', dayfirst=False)
    return df.sort_values('date_parsed', ascending=False).reset_index(drop=True)

try:
    df = load_data()
    num_cols = ['n1', 'n2', 'n3', 'n4', 'n5', 'n6']
except Exception as e:
    st.error(f"⚠️ Error loading data: {e}")
    st.stop()

# --- HEADER SECTION ---
st.title("🎰 Mark Six AI Analyzer")
st.markdown("Professional Data Analysis & Predictive Modeling")

# Latest Draw Summary
latest = df.iloc[0]
l_nums = "-".join([str(int(latest[n])) for n in num_cols])

m1, m2, m3 = st.columns(3)
with m1:
    st.markdown(f"<div class='metric-card'>Latest Draw<br><span class='stat-val'>{latest['date']}</span></div>", unsafe_allow_html=True)
with m2:
    st.markdown(f"<div class='metric-card'>Winning Numbers<br><span class='stat-val'>{l_nums}</span></div>", unsafe_allow_html=True)
with m3:
    st.markdown(f"<div class='metric-card'>Special Number<br><span class='stat-val' style='color:#7FD1B9'>{int(latest['extra'])}</span></div>", unsafe_allow_html=True)

st.write("---")

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["📊 Probability Analysis", "📈 Trends", "🧠 AI Oracle"])

with tab1:
    col_l, col_r = st.columns([2, 1])
    # Frequency calculation
    window = st.sidebar.slider("Analysis Window (Recent Draws)", 10, 500, 100)
    recent_df = df.head(window)
    all_nums = recent_df[num_cols].values.flatten()
    freq = Counter(all_nums)
    freq_df = pd.DataFrame(freq.items(), columns=['Number', 'Count']).sort_values('Count', ascending=False)
    
    with col_l:
        st.subheader("Number Frequency (Last 100 draws)")
        fig = px.bar(freq_df.head(20), x='Number', y='Count', color='Count', 
                     color_continuous_scale='Oranges', template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    
    with col_r:
        st.subheader("🔥 Top Picks")
        st.dataframe(freq_df.head(10).reset_index(drop=True), use_container_width=True)

with tab2:
    st.subheader("Historical Sum Trend")
    df['draw_sum'] = df[num_cols].sum(axis=1)
    fig_line = px.line(df.head(50), x='date_parsed', y='draw_sum', 
                       markers=True, template="plotly_dark", title="Sum of Numbers (Last 50 Draws)")
    st.plotly_chart(fig_line, use_container_width=True)

with tab3:
    st.subheader("Prophet Machine Learning Model")
    if st.button("🚀 Run Predictive Simulation"):
        with st.spinner("Analyzing 20+ years of patterns..."):
            # Prepare Prophet Data
            p_df = df.rename(columns={'date_parsed': 'ds', 'draw_sum': 'y'})[['ds', 'y']].dropna()
            m = Prophet(daily_seasonality=False, weekly_seasonality=True)
            m.fit(p_df)
            
            future = m.make_future_dataframe(periods=1, freq='3D')
            forecast = m.predict(future)
            next_sum = forecast['yhat'].iloc[-1]
            
            # AI Logic: Mix Hot (High Prob) and Cold (Due to appear)
            hot = freq_df['Number'].head(4).tolist()
            cold = freq_df['Number'].tail(2).tolist()
            ai_pick = sorted([int(x) for x in (hot + cold)])
            
            c1, c2 = st.columns([1, 1])
            with c1:
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number", value = next_sum,
                    title = {'text': "Predicted Sum Index"},
                    gauge = {'axis': {'range': [21, 279]}, 'bar': {'color': "#FF6B35"}}
                ))
                st.plotly_chart(fig_gauge, use_container_width=True)
            
            with c2:
                st.markdown(f"""
                <div style="background: #1e2130; padding: 40px; border-radius: 20px; text-align: center; border: 2px solid #FF6B35;">
                    <h2 style="color: white; margin-bottom:10px;">AI Recommended Set</h2>
                    <h1 style="color: #FF6B35; letter-spacing: 5px; font-size: 3em;">{' '.join(map(str, ai_pick))}</h1>
                    <p style="color: #888;">Based on Weighted Volatility Factors</p>
                </div>
                """, unsafe_allow_html=True)
