import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from collections import Counter

# --- CONFIGURATION ---
st.set_page_config(page_title="Mark Six Pro AI", page_icon="🎰", layout="wide")

# Modern Professional Styling
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    div.stButton > button:first-child {
        background: linear-gradient(45deg, #FF6B35, #FF8C5A);
        color: white; border: none; border-radius: 10px; width: 100%; height: 3.5em; font-weight: bold;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        padding: 20px; border-radius: 15px; border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
    }
    .stat-val { color: #FF6B35; font-size: 28px; font-weight: bold; }
    .stat-label { color: #888; font-size: 14px; text-transform: uppercase; letter-spacing: 1px; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_data():
    # Load and handle mixed date formats (YYYY/MM/DD and DD/MM/YYYY)
    df = pd.read_csv('marksix.csv')
    df['date_parsed'] = pd.to_datetime(df['date'], format='mixed')
    return df.sort_values('date_parsed', ascending=False).reset_index(drop=True)

try:
    df = load_data()
    total_records = len(df)
    num_cols = ['n1', 'n2', 'n3', 'n4', 'n5', 'n6']
except Exception as e:
    st.error(f"⚠️ Error loading data: {e}")
    st.stop()

# --- SIDEBAR CONTROL PANEL ---
with st.sidebar:
    st.title("Settings")
    # Dynamically set max to the length of the CSV
    window = st.slider(
        "Analysis Window (Draws)", 
        min_value=10, 
        max_value=total_records, 
        value=min(100, total_records),
        help="Select how many recent draws to include in the statistical analysis."
    )
    st.divider()
    st.info(f"📊 Dataset: {total_records} total draws.")

# --- HEADER SECTION ---
st.title("🎰 Mark Six AI Analyzer")
latest = df.iloc[0]
l_nums = "  ".join([str(int(latest[n])) for n in num_cols])

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f"<div class='metric-card'><span class='stat-label'>Latest Draw Date</span><br><span class='stat-val'>{latest['date']}</span></div>", unsafe_allow_html=True)
with c2:
    st.markdown(f"<div class='metric-card'><span class='stat-label'>Main Numbers</span><br><span class='stat-val'>{l_nums}</span></div>", unsafe_allow_html=True)
with c3:
    st.markdown(f"<div class='metric-card'><span class='stat-label'>Special Number</span><br><span class='stat-val' style='color:#7FD1B9'>{int(latest['extra'])}</span></div>", unsafe_allow_html=True)

st.write("---")

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["📊 Statistical Insights", "📈 Trend Reports", "🧠 AI Oracle"])

with tab1:
    col_l, col_r = st.columns([2, 1])
    
    # Analyze based on the slider window
    recent_df = df.head(window)
    all_draws = recent_df[num_cols].values.flatten()
    freq = Counter(all_draws)
    freq_df = pd.DataFrame(freq.items(), columns=['Number', 'Count']).sort_values('Count', ascending=False)
    
    with col_l:
        st.subheader(f"Frequency Distribution (Last {window} Draws)")
        fig = px.bar(freq_df.head(25), x='Number', y='Count', 
                     color='Count', color_continuous_scale='Oranges', 
                     template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    
    with col_r:
        st.subheader("🔥 Top 10 Hot Numbers")
        st.dataframe(freq_df.head(10).reset_index(drop=True), use_container_width=True)
        st.subheader("❄️ Top 5 Cold Numbers")
        st.dataframe(freq_df.tail(5).reset_index(drop=True), use_container_width=True)

with tab2:
    st.subheader("Historical Summation Trend")
    df['draw_sum'] = df[num_cols].sum(axis=1)
    fig_trend = px.area(df.head(window), x='date_parsed', y='draw_sum', 
                        title=f"Sum of Winning Numbers (Last {window} draws)",
                        template="plotly_dark", color_discrete_sequence=['#FF6B35'])
    st.plotly_chart(fig_trend, use_container_width=True)

with tab3:
    st.header("Prophet Forecasting Engine")
    st.write("This engine uses Facebook's Prophet model to analyze time-series patterns in the summation of winning numbers.")
    
    if st.button("Generate AI Predictive Report"):
        with st.spinner("Training model on historical dataset..."):
            # Prepare Prophet Data
            p_df = df.rename(columns={'date_parsed': 'ds', 'draw_sum': 'y'})[['ds', 'y']].dropna()
            m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
            m.fit(p_df)
            
            # Predict 1 step ahead (next draw)
            future = m.make_future_dataframe(periods=1, freq='3D')
            forecast = m.predict(future)
            next_sum = forecast['yhat'].iloc[-1]
            
            # Smart AI Logic: Balance Hot and Cold
            hot_nums = freq_df['Number'].head(4).tolist()
            cold_nums = freq_df['Number'].tail(2).tolist()
            ai_suggestion = sorted([int(x) for x in (hot_nums + cold_nums)])
            
            c1, c2 = st.columns([1, 1])
            with c1:
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number", value = next_sum,
                    title = {'text': "Predicted Next Draw Sum"},
                    gauge = {'axis': {'range': [21, 279]}, 'bar': {'color': "#FF6B35"}}
                ))
                fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
                st.plotly_chart(fig_gauge, use_container_width=True)
            
            with c2:
                st.markdown(f"""
                <div style="background: rgba(255, 107, 53, 0.1); padding: 40px; border-radius: 20px; text-align: center; border: 2px solid #FF6B35; margin-top: 20px;">
                    <h3 style="color: white; margin-bottom:10px;">AI Recommended Combo</h3>
                    <h1 style="color: #FF6B35; letter-spacing: 8px; font-size: 3.5em;">{' '.join(map(str, ai_suggestion))}</h1>
                    <p style="color: #888;">Optimized using Weighted Frequency (Hot/Cold) balance.</p>
                </div>
                """, unsafe_allow_html=True)
