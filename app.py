import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from collections import Counter
import datetime

# --- CONFIGURATION ---
st.set_page_config(page_title="Mark Six Pro AI", page_icon="🎰", layout="wide")

# Custom Professional CSS
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    div.stButton > button:first-child {
        background: linear-gradient(45deg, #FF6B35, #FF8C5A);
        color: white; border: none; border-radius: 10px; width: 100%;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        padding: 20px; border-radius: 15px; border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .stat-val { color: #FF6B35; font-size: 24px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_data():
    df = pd.read_csv('marksix.csv')
    df['date_parsed'] = pd.to_datetime(df['date'], format='%d/%m/%Y', dayfirst=True)
    return df.sort_values('date_parsed', ascending=False)

try:
    df = load_data()
    num_cols = ['n1', 'n2', 'n3', 'n4', 'n5', 'n6']
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# --- SIDEBAR & HEADER ---
with st.sidebar:
    st.image("https://www.flaticon.com/free-icons/lottery", width=100) # Optional placeholder
    st.title("Control Panel")
    analysis_range = st.slider("Analysis Window (Draws)", 10, 500, 100)
    st.info(f"Analyzing data from {df['date'].iloc[-1]} to {df['date'].iloc[0]}")

st.title("🎰 Mark Six AI Analyzer")
st.markdown("---")

# --- TOP METRICS ---
l_date = df.iloc[0]['date']
l_nums = "-".join([str(int(df.iloc[0][n])) for n in num_cols])
s_num = int(df.iloc[0]['extra'])

m1, m2, m3 = st.columns(3)
with m1:
    st.markdown(f"<div class='metric-card'>Latest Draw Date<br><span class='stat-val'>{l_date}</span></div>", unsafe_allow_html=True)
with m2:
    st.markdown(f"<div class='metric-card'>Winning Numbers<br><span class='stat-val'>{l_nums}</span></div>", unsafe_allow_html=True)
with m3:
    st.markdown(f"<div class='metric-card'>Special Number<br><span class='stat-val' style='color:#7FD1B9'>{s_num}</span></div>", unsafe_allow_html=True)

st.write("")

# --- TABS ---
tabs = st.tabs(["📊 Frequency Analysis", "📈 Predictive Trends", "🤖 AI Oracle"])

with tabs[0]:
    col_l, col_r = st.columns([2, 1])
    
    # Calculate Frequency
    recent_df = df.head(analysis_range)
    all_draws = recent_df[num_cols].values.flatten()
    counts = Counter(all_draws)
    freq_df = pd.DataFrame(counts.items(), columns=['Number', 'Frequency']).sort_values('Frequency', ascending=False)
    
    with col_l:
        st.subheader("Number Distribution")
        fig = px.bar(freq_df.head(20), x='Number', y='Frequency', 
                     color='Frequency', color_continuous_scale='Oranges',
                     template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
        
    with col_r:
        st.subheader("🔥 Hot vs ❄️ Cold")
        st.write("Top 5 Hot Numbers")
        st.dataframe(freq_df.head(5), hide_index=True)
        st.write("Top 5 Cold Numbers")
        st.dataframe(freq_df.tail(5), hide_index=True)

with tabs[1]:
    st.subheader("Draw Sum Trend Analysis")
    df['draw_sum'] = df[num_cols].sum(axis=1)
    fig_line = px.line(df.head(100), x='date_parsed', y='draw_sum', 
                      title="Sum of Numbers Over Time", template="plotly_dark")
    fig_line.add_hline(y=150, line_dash="dash", line_color="red", annotation_text="High Average")
    st.plotly_chart(fig_line, use_container_width=True)

with tabs[2]:
    st.header("Prophet Forecasting Engine")
    if st.button("Generate AI Intelligence Report"):
        with st.spinner("Crunching historical patterns..."):
            # Prepare Prophet Data
            p_df = df.rename(columns={'date_parsed': 'ds', 'draw_sum': 'y'})[['ds', 'y']]
            model = Prophet(changepoint_prior_scale=0.05, daily_seasonality=False)
            model.fit(p_df)
            
            future = model.make_future_dataframe(periods=5, freq='3D')
            forecast = model.predict(future)
            
            # Smart Logic for AI Pick
            hot_numbers = [int(x) for x in freq_df['Number'].head(4).tolist()]
            cold_numbers = [int(x) for x in freq_df['Number'].tail(2).tolist()]
            ai_suggestion = sorted(hot_numbers + cold_numbers)
            
            st.success("Analysis Complete")
            
            c1, c2 = st.columns([1, 1])
            with c1:
                # Gauge Chart for expected sum
                expected_sum = forecast['yhat'].iloc[-1]
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number", value = expected_sum,
                    title = {'text': "Predicted Next Sum"},
                    gauge = {'axis': {'range': [21, 279]}, 'bar': {'color': "#FF6B35"}}
                ))
                st.plotly_chart(fig_gauge, use_container_width=True)
                
            with c2:
                st.markdown(f"""
                <div style="background: #1e2130; padding: 40px; border-radius: 20px; text-align: center; border: 2px solid #FF6B35;">
                    <h3 style="color: white;">AI Recommended Combination</h3>
                    <h1 style="color: #FF6B35; letter-spacing: 5px;">{' '.join(map(str, ai_suggestion))}</h1>
                    <p style="color: #888;">Based on 4 'Hot' and 2 'Cold' volatility factors</p>
                </div>
                """, unsafe_allow_html=True)
