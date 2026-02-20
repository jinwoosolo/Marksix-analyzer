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
    .win-tag { padding: 4px 8px; border-radius: 5px; font-weight: bold; color: white; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_data():
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
    window = st.slider(
        "Analysis Window (Draws)", 
        min_value=10, 
        max_value=total_records, 
        value=min(100, total_records),
        help="Select how many recent draws to include in statistics."
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
tab1, tab2, tab3, tab4 = st.tabs(["📊 Statistics", "📈 Trends", "🧠 AI Oracle", "🔍 Historical Checker"])

with tab1:
    col_l, col_r = st.columns([2, 1])
    recent_df = df.head(window)
    all_draws = recent_df[num_cols].values.flatten()
    freq = Counter(all_draws)
    freq_df = pd.DataFrame(freq.items(), columns=['Number', 'Count']).sort_values('Count', ascending=False)
    
    with col_l:
        st.subheader(f"Frequency (Last {window} Draws)")
        fig = px.bar(freq_df.head(25), x='Number', y='Count', color='Count', color_continuous_scale='Oranges', template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    with col_r:
        st.subheader("🔥 Top 10 Hot")
        st.dataframe(freq_df.head(10).reset_index(drop=True), use_container_width=True)

with tab2:
    st.subheader("Historical Summation Trend")
    df['draw_sum'] = df[num_cols].sum(axis=1)
    fig_trend = px.area(df.head(window), x='date_parsed', y='draw_sum', template="plotly_dark", color_discrete_sequence=['#FF6B35'])
    st.plotly_chart(fig_trend, use_container_width=True)

with tab3:
    st.header("Prophet Forecasting Engine")
    if st.button("Generate AI Predictive Report"):
        with st.spinner("Analyzing historical dataset..."):
            p_df = df.rename(columns={'date_parsed': 'ds', 'draw_sum': 'y'})[['ds', 'y']].dropna()
            m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
            m.fit(p_df)
            future = m.make_future_dataframe(periods=1, freq='3D')
            forecast = m.predict(future)
            next_sum = forecast['yhat'].iloc[-1]
            hot_nums = freq_df['Number'].head(4).tolist()
            cold_nums = freq_df['Number'].tail(2).tolist()
            ai_suggestion = sorted([int(x) for x in (hot_nums + cold_nums)])
            
            c1, c2 = st.columns([1, 1])
            with c1:
                fig_gauge = go.Figure(go.Indicator(mode="gauge+number", value=next_sum, title={'text': "Predicted Next Sum"}, gauge={'axis': {'range': [21, 279]}, 'bar': {'color': "#FF6B35"}}))
                fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
                st.plotly_chart(fig_gauge, use_container_width=True)
            with c2:
                st.markdown(f"<div style='background:rgba(255,107,53,0.1);padding:40px;border-radius:20px;text-align:center;border:2px solid #FF6B35;margin-top:20px;'><h3 style='color:white;'>AI Recommended Combo</h3><h1 style='color:#FF6B35;letter-spacing:8px;font-size:3.5em;'>{' '.join(map(str, ai_suggestion))}</h1></div>", unsafe_allow_html=True)

with tab4:
    st.header("🔎 Historical Prize Checker")
    st.write("Select 6 numbers to see if you would have won any prizes in the past.")
    
    # Input numbers
    selected_nums = st.multiselect("Pick 6 Numbers:", options=list(range(1, 50)), max_selections=6)
    
    if len(selected_nums) == 6:
        user_set = set(selected_nums)
        win_records = []
        
        for idx, row in df.iterrows():
            draw_set = set([row['n1'], row['n2'], row['n3'], row['n4'], row['n5'], row['n6']])
            extra_num = row['extra']
            
            matched = user_set.intersection(draw_set)
            match_count = len(matched)
            has_extra = extra_num in user_set
            
            prize = ""
            if match_count == 6: prize = "1st Prize"
            elif match_count == 5 and has_extra: prize = "2nd Prize"
            elif match_count == 5: prize = "3rd Prize"
            elif match_count == 4 and has_extra: prize = "4th Prize"
            elif match_count == 4: prize = "5th Prize"
            elif match_count == 3 and has_extra: prize = "6th Prize"
            elif match_count == 3: prize = "7th Prize"
            
            if prize:
                win_records.append({
                    "Date": row['date'],
                    "Prize": prize,
                    "Matched": sorted(list(matched)),
                    "Draw Numbers": f"{row['n1']}, {row['n2']}, {row['n3']}, {row['n4']}, {row['n5']}, {row['n6']} + ({row['extra']})"
                })
        
        if win_records:
            st.success(f"🎉 Your numbers have won **{len(win_records)}** times in history!")
            win_df = pd.DataFrame(win_records)
            
            # Styling the prize display
            def color_prize(val):
                color = '#FF6B35' if '1st' in val or '2nd' in val else '#888'
                return f'color: {color}; font-weight: bold'

            st.dataframe(win_df.style.applymap(color_prize, subset=['Prize']), use_container_width=True, hide_index=True)
        else:
            st.info("No prizes won with these numbers yet. Keep trying!")
    elif len(selected_nums) > 0:
        st.warning(f"Please select exactly 6 numbers (you have selected {len(selected_nums)}).")
