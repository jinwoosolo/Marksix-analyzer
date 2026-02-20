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
        background: rgba(255, 255, 255, 0.05);
        color: white; border: 1px solid rgba(255, 255, 255, 0.2); border-radius: 8px;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover { border-color: #FF6B35; color: #FF6B35; }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        padding: 20px; border-radius: 15px; border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center; margin-bottom: 10px;
    }
    .stat-val { color: #FF6B35; font-size: 28px; font-weight: bold; }
    .stat-label { color: #888; font-size: 14px; text-transform: uppercase; letter-spacing: 1px; }
    .favorite-card {
        background: rgba(255, 255, 255, 0.03);
        padding: 12px; border-radius: 12px; border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 5px;
    }
    .win-highlight { color: #FF6B35; font-weight: bold; font-size: 1.1em; }
    .date-list { font-size: 0.8em; color: #aaa; margin-top: 5px; line-height: 1.4; }
    .section-header { margin-top: 0px; margin-bottom: 10px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- PERSISTENCE LOGIC (QUERY PARAMS) ---
def sync_favs_to_url():
    """Update URL query parameters to match the current fav_sets in session state."""
    params = {}
    for i, fav in enumerate(st.session_state.fav_sets):
        if fav:
            params[f"fav{i}"] = ",".join(map(str, sorted(list(fav))))
    st.query_params.update(params)

def load_favs_from_url():
    """Initialize session state from URL query parameters."""
    favs = [None, None, None]
    for i in range(3):
        param_val = st.query_params.get(f"fav{i}")
        if param_val:
            try:
                favs[i] = [int(x) for x in param_val.split(",")]
            except:
                pass
    return favs

# --- SESSION STATE ---
if 'selected_nums' not in st.session_state:
    st.session_state.selected_nums = set()
if 'fav_sets' not in st.session_state:
    st.session_state.fav_sets = load_favs_from_url()

@st.cache_data(ttl=3600)
def load_data():
    df = pd.read_csv('marksix.csv')
    df['date_parsed'] = pd.to_datetime(df['date'], format='mixed')
    return df.sort_values('date_parsed', ascending=False).reset_index(drop=True)

def calculate_prizes(user_set, data):
    """Calculates all prizes for a given set across the dataset."""
    user_set = set(user_set)
    results = []
    for _, row in data.iterrows():
        draw_set = {row['n1'], row['n2'], row['n3'], row['n4'], row['n5'], row['n6']}
        extra_num = row['extra']
        matched = user_set.intersection(draw_set)
        match_count = len(matched)
        has_extra = extra_num in user_set
        
        prize = ""
        rank = 99
        if match_count == 6: prize, rank = "1st Prize", 1
        elif match_count == 5 and has_extra: prize, rank = "2nd Prize", 2
        elif match_count == 5: prize, rank = "3rd Prize", 3
        elif match_count == 4 and has_extra: prize, rank = "4th Prize", 4
        elif match_count == 4: prize, rank = "5th Prize", 5
        elif match_count == 3 and has_extra: prize, rank = "6th Prize", 6
        elif match_count == 3: prize, rank = "7th Prize", 7
            
        if prize:
            results.append({"Date": row['date'], "Prize": prize, "Rank": rank, "Match": match_count})
    return results

try:
    df = load_data()
    total_records = len(df)
    num_cols = ['n1', 'n2', 'n3', 'n4', 'n5', 'n6']
except Exception as e:
    st.error(f"⚠️ Error loading data: {e}")
    st.stop()

# --- SIDEBAR CONTROL PANEL ---
with st.sidebar:
    st.title("Control Panel")
    
    st.subheader("Visibility Settings")
    show_favs = st.checkbox("Show Favorite Tracker", value=True)
    show_checker = st.checkbox("Show Prize Checker", value=True)
    show_analysis = st.checkbox("Show Analysis Tabs", value=True)
    
    st.divider()
    
    if show_analysis:
        st.subheader("Analysis Parameters")
        window = st.slider("Statistics Window", 10, total_records, 100)
    
    st.divider()
    st.info(f"📊 Total Draws: {total_records}")

# --- HEADER SECTION ---
st.title("🎰 Mark Six AI Analyzer")
latest = df.iloc[0]
l_nums = "  ".join([str(int(latest[n])) for n in num_cols])

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f"<div class='metric-card'><span class='stat-label'>Latest Draw Date</span><br><span class='stat-val'>{latest['date']}</span></div>", unsafe_allow_html=True)
with c2:
    st.markdown(f"<div class='metric-card'><span class='stat-label'>Numbers</span><br><span class='stat-val'>{l_nums}</span></div>", unsafe_allow_html=True)
with c3:
    st.markdown(f"<div class='metric-card'><span class='stat-label'>Special</span><br><span class='stat-val' style='color:#7FD1B9'>{int(latest['extra'])}</span></div>", unsafe_allow_html=True)

# --- 1. FAVORITE SETS TRACKER ---
if show_favs:
    st.markdown("<h3 class='section-header'>⭐ Favorite Sets Tracker</h3>", unsafe_allow_html=True)
    fav_cols = st.columns(3)
    for idx, fav in enumerate(st.session_state.fav_sets):
        with fav_cols[idx]:
            if fav:
                st.markdown(f"<div class='favorite-card'>", unsafe_allow_html=True)
                st.markdown(f"**Slot {idx+1}**")
                st.code(f"{' '.join(map(str, fav))}")
                fav_results = calculate_prizes(fav, df)
                high_tier = [r for r in fav_results if r['Rank'] <= 4]
                latest_draw_set = {latest['n1'], latest['n2'], latest['n3'], latest['n4'], latest['n5'], latest['n6']}
                latest_match = len(set(fav).intersection(latest_draw_set))
                
                st.markdown(f"Total Wins: **{len(fav_results)}**")
                st.markdown(f"Major Wins (4th+): <span class='win-highlight'>{len(high_tier)}</span>", unsafe_allow_html=True)
                
                if high_tier:
                    dates = [r['Date'] for r in high_tier]
                    st.markdown(f"<div class='date-list'><b>Major Win Dates:</b><br>{', '.join(dates)}</div>", unsafe_allow_html=True)
                
                if latest_match >= 3:
                    st.warning(f"🔔 Latest: {latest_match} matches!")
                else:
                    st.write(f"Latest: {latest_match} matches")
                    
                if st.button(f"Clear Slot {idx+1}", key=f"clear_{idx}", use_container_width=True):
                    st.session_state.fav_sets[idx] = None
                    sync_favs_to_url()
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.info(f"Slot {idx+1} Empty")

# --- 2. HISTORICAL PRIZE CHECKER ---
if show_checker:
    if show_favs:
        st.write("---")
    header_col1, header_col2 = st.columns([5, 1])
    with header_col1:
        st.markdown("<h3 class='section-header'>🔎 Historical Prize Checker</h3>", unsafe_allow_html=True)
    with header_col2:
        if st.button("Reset Selection", use_container_width=True):
            st.session_state.selected_nums = set()
            st.rerun()

    st.write("Click numbers to select your 6-number combination.")

    # Grid UI
    grid_cols = st.columns(7)
    for i in range(1, 49 + 1):
        with grid_cols[(i-1) % 7]:
            is_selected = i in st.session_state.selected_nums
            if st.button(f"{i:02d}", key=f"btn_{i}", use_container_width=True, 
                         type="primary" if is_selected else "secondary"):
                if i in st.session_state.selected_nums:
                    st.session_state.selected_nums.remove(i)
                elif len(st.session_state.selected_nums) < 6:
                    st.session_state.selected_nums.add(i)
                st.rerun()

    selected_list = sorted(list(st.session_state.selected_nums))
    st.markdown(f"**Selected:** `{', '.join(map(str, selected_list)) if selected_list else 'None'}` ({len(selected_list)}/6)")

    if len(selected_list) == 6:
        st.write("💾 **Save to Favorites:**")
        save_cols = st.columns(3)
        for idx in range(3):
            with save_cols[idx]:
                if st.button(f"Save to Slot {idx+1}", key=f"save_{idx}", use_container_width=True):
                    st.session_state.fav_sets[idx] = selected_list
                    sync_favs_to_url()
                    st.toast(f"Slot {idx+1} saved!")
                    st.rerun()

        results = calculate_prizes(selected_list, df)
        if results:
            st.success(f"🎉 Result found: **{len(results)}** wins in total")
            results_df = pd.DataFrame(results).sort_values(by="Rank", ascending=True)
            st.dataframe(results_df[["Date", "Prize"]], use_container_width=True, hide_index=True)
        else:
            st.info("No historical wins found for this set.")

# --- ANALYSIS TABS ---
if show_analysis:
    if show_favs or show_checker:
        st.write("---")
    
    tab1, tab2, tab3 = st.tabs(["📊 Statistics", "📈 Trends", "🧠 AI Oracle"])

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
            st.subheader("🔥 Top Hot")
            st.dataframe(freq_df.head(10).reset_index(drop=True), use_container_width=True)

    with tab2:
        st.subheader("Historical Summation Trend")
        df['draw_sum'] = df[num_cols].sum(axis=1)
        fig_trend = px.area(df.head(window), x='date_parsed', y='draw_sum', template="plotly_dark", color_discrete_sequence=['#FF6B35'])
        st.plotly_chart(fig_trend, use_container_width=True)

    with tab3:
        st.header("Prophet Forecasting Engine")
        if st.button("Generate AI Predictive Report"):
            with st.spinner("Analyzing patterns..."):
                p_df = df.rename(columns={'date_parsed': 'ds', 'draw_sum': 'y'})[['ds', 'y']].dropna()
                m = Prophet(daily_seasonality=False, weekly_seasonality=True)
                m.fit(p_df)
                future = m.make_future_dataframe(periods=1, freq='3D')
                forecast = m.predict(future)
                next_sum = forecast['yhat'].iloc[-1]
                c1, c2 = st.columns(2)
                with c1:
                    fig_gauge = go.Figure(go.Indicator(mode="gauge+number", value=next_sum, title={'text': "Predicted Sum Index"}, gauge={'bar': {'color': "#FF6B35"}}))
                    st.plotly_chart(fig_gauge, use_container_width=True)
                with c2:
                    hot = freq_df['Number'].head(4).tolist()
                    cold = freq_df['Number'].tail(2).tolist()
                    ai_suggestion = sorted([int(x) for x in (hot + cold)])
                    st.markdown(f"<div style='background:rgba(255,107,53,0.1);padding:40px;border-radius:20px;text-align:center;border:2px solid #FF6B35;'><h3 style='color:white;'>AI Recommended</h3><h1 style='color:#FF6B35;letter-spacing:8px;'>{' '.join(map(str, ai_suggestion))}</h1></div>", unsafe_allow_html=True)
