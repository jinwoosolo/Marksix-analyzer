import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from collections import Counter
import datetime
import numpy as np

# --- APP 配置 ---
st.set_page_config(page_title="六合彩 AI 專業分析器 Pro", page_icon="🎰", layout="wide")

# 專業介面 CSS 樣式
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
        background: rgba(255, 255, 255, 0.05); padding: 20px; border-radius: 15px; 
        border: 1px solid rgba(255, 255, 255, 0.1); text-align: center; margin-bottom: 10px;
    }
    .stat-val { color: #FF6B35; font-size: 28px; font-weight: bold; }
    .stat-label { color: #888; font-size: 14px; text-transform: uppercase; letter-spacing: 1px; }
    .favorite-card {
        background: rgba(255, 255, 255, 0.03); padding: 12px; border-radius: 12px; 
        border: 1px solid rgba(255, 255, 255, 0.1); margin-bottom: 5px;
    }
    .win-highlight { color: #FF6B35; font-weight: bold; font-size: 1.1em; }
    .section-header { margin-top: 15px; margin-bottom: 15px; font-weight: bold; border-left: 5px solid #FF6B35; padding-left: 10px; }
    .ai-box { background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 107, 53, 0.3); border-radius: 15px; padding: 20px; margin-bottom: 20px; }
    .history-card { background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.05); padding: 15px; border-radius: 10px; margin-bottom: 10px; }
    .match-tag { background: #FF6B35; color: white; padding: 2px 8px; border-radius: 4px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- 固定最愛組合 ---
FIXED_FAV_SETS = [
    {5, 11, 12, 13, 15, 27},
    {9, 10, 22, 24, 30, 49},
    {19, 23, 25, 39, 44, 46}
]

# --- 核心數據邏輯 ---
@st.cache_data(ttl=3600)
def load_data():
    df = pd.read_csv('marksix.csv')
    df['date_parsed'] = pd.to_datetime(df['date'], format='mixed')
    df = df.sort_values('date_parsed', ascending=True).reset_index(drop=True)
    draws_matrix = df[['n1', 'n2', 'n3', 'n4', 'n5', 'n6']].values
    extras_array = df['extra'].values
    dates_array = df['date'].values
    return df, draws_matrix, extras_array, dates_array

def calculate_ai_scores(historical_df, window_size=100):
    """根據頻率與遺漏值計算每個號碼的綜合機率得分，並按得分排序"""
    recent_data = historical_df.tail(window_size)
    nums = recent_data[['n1', 'n2', 'n3', 'n4', 'n5', 'n6']].values.flatten()
    counts = Counter(nums)
    scores = {}
    total_draws = len(recent_data)
    for n in range(1, 50):
        # 1. 頻率權重 (佔 60%)
        freq = counts.get(n, 0)
        freq_score = (freq / (total_draws * 6 / 49)) * 60 
        
        # 2. 遺漏權重 (佔 40%) - 越久沒開代表回歸機率調整
        last_appearance = 0
        for i, draw in enumerate(reversed(recent_data[['n1', 'n2', 'n3', 'n4', 'n5', 'n6']].values)):
            if n in draw:
                last_appearance = i
                break
        gap_score = (last_appearance / 20) * 40 # 假設 20 期為一個觀察週期
        
        scores[n] = freq_score + gap_score
        
    # 按得分由高到低排列（最大機會開出的數字排在前面）
    ranked_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [int(x[0]) for x in ranked_nums[:12]]

@st.cache_data(show_spinner=False)
def get_historical_backtest(limit=100):
    """回測最近 100 期的 AI 表現"""
    df_asc, draws_np, extras_np, dates_np = load_data()
    results = []
    
    # 從最後一期往回推
    total = len(df_asc)
    start_idx = total - limit
    
    for i in range(total - 1, start_idx - 1, -1):
        if i < 50: continue 
        
        # 模擬當時預測：只看當期之前的數據
        past_df = df_asc.iloc[:i]
        target_draw = set(draws_np[i])
        target_extra = int(extras_np[i])
        
        # AI 預測（按得分降序排）
        ai_12 = calculate_ai_scores(past_df, window_size=100)
        
        # 命中計算 (包含特別號)
        matched_nums = [n for n in ai_12 if n in target_draw]
        matched_extra = target_extra in ai_12
        match_count = len(matched_nums) + (1 if matched_extra else 0)
        
        results.append({
            "date": dates_np[i],
            "draw_nums": sorted(list(target_draw)),
            "extra": target_extra,
            "prediction": ai_12, # 這是按機率排的
            "match_count": match_count,
            "matched_list": matched_nums + ([target_extra] if matched_extra else [])
        })
    return results

@st.cache_data(show_spinner=False)
def get_all_historical_wins_fast(user_set_tuple, total_count):
    _, draws, extras, dates = load_data()
    user_set = set(user_set_tuple)
    results = []
    for i in range(len(draws)):
        matched = len(user_set.intersection(set(draws[i])))
        if matched < 3: continue
        has_extra = int(extras[i]) in user_set
        prize, rank = None, 99
        if matched == 6: prize, rank = "1st Prize", 1
        elif matched == 5 and has_extra: prize, rank = "2nd Prize", 2
        elif matched == 5: prize, rank = "3rd Prize", 3
        elif matched == 4 and has_extra: prize, rank = "4th Prize", 4
        elif matched == 4: prize, rank = "5th Prize", 5
        elif matched == 3 and has_extra: prize, rank = "6th Prize", 6
        elif matched == 3: prize, rank = "7th Prize", 7
        if prize: results.append({"Date": dates[i], "Prize": prize, "Rank": rank})
    return sorted(results, key=lambda x: (x['Rank'], x['Date']), reverse=False)

# --- 執行載入 ---
try:
    df_asc, draws_np, extras_np, dates_np = load_data()
    df_desc = df_asc.sort_values('date_parsed', ascending=False).reset_index(drop=True)
    total_records = len(df_asc)
    num_cols = ['n1', 'n2', 'n3', 'n4', 'n5', 'n6']
except Exception as e:
    st.error(f"數據載入出錯: {e}"); st.stop()

if 'selected_nums' not in st.session_state: st.session_state.selected_nums = set()

# --- 側邊欄 ---
with st.sidebar:
    st.title("⚙️ 控制面板")
    v_fav = st.checkbox("顯示組合追蹤", value=True)
    v_ai = st.checkbox("顯示 AI 預測與回測", value=True)
    v_check = st.checkbox("顯示中獎檢查器", value=False)
    v_chart = st.checkbox("顯示分析圖表", value=False)
    st.divider()
    window = st.slider("預測參考窗口", 50, 500, 100)

# --- 頁首資訊 ---
latest = df_desc.iloc[0]
st.title("🎰 六合彩 AI 專業分析器 Pro")
c_m1, c_m2, c_m3 = st.columns(3)
with c_m1: st.markdown(f"<div class='metric-card'><span class='stat-label'>最近日期</span><br><span class='stat-val'>{latest['date']}</span></div>", unsafe_allow_html=True)
with c_m2: st.markdown(f"<div class='metric-card'><span class='stat-label'>中獎號碼</span><br><span class='stat-val'>{'  '.join([str(int(latest[n])) for n in num_cols])}</span></div>", unsafe_allow_html=True)
with c_m3: st.markdown(f"<div class='metric-card'><span class='stat-label'>特別號碼</span><br><span class='stat-val' style='color:#7FD1B9'>{int(latest['extra'])}</span></div>", unsafe_allow_html=True)

# --- 1. 固定組合即時追蹤 ---
if v_fav:
    st.markdown("<h3 class='section-header'>⭐ 固定組合即時追蹤</h3>", unsafe_allow_html=True)
    f_cols = st.columns(3)
    for i, fav in enumerate(FIXED_FAV_SETS):
        with f_cols[i]:
            st.markdown("<div class='favorite-card'>", unsafe_allow_html=True)
            st.markdown(f"**固定組合 {i+1}**")
            sorted_fav = sorted(list(fav))
            st.code(" ".join(map(str, sorted_fav)))
            f_res = get_all_historical_wins_fast(tuple(sorted_fav), total_records)
            high = [r for r in f_res if r['Rank'] <= 4]
            st.markdown(f"總中獎: **{len(f_res)}** | 大獎: <span class='win-highlight'>{len(high)}</span>", unsafe_allow_html=True)
            if high:
                formatted_wins = [f"{r['Date']} ({r['Prize']})" for r in high[:3]]
                st.markdown(f"<div class='date-list'><b>近期大獎:</b><br>{'<br>'.join(formatted_wins)}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

# --- 2. AI 智能預測與歷史回測 ---
if v_ai:
    st.write("---")
    st.markdown("<h3 class='section-header'>🔮 AI 智能預測與深度分析 (下期推薦)</h3>", unsafe_allow_html=True)
    
    # 計算下期預測 (按機率得分排序)
    next_ai_12 = calculate_ai_scores(df_asc, window_size=window)
    
    st.markdown("<div class='ai-box'>", unsafe_allow_html=True)
    a_c1, a_c2 = st.columns([1, 1])
    with a_c1:
        st.markdown("#### 🎯 下期推薦 12 字 (按機率由高至低排序)")
        st.markdown(f"""<div style='background: linear-gradient(45deg, #FF6B35, #F7931E); padding: 25px; border-radius: 15px; text-align: center; color: white;'><h2 style='letter-spacing: 3px;'>{' '.join(map(str, next_ai_12[:6]))}<br>{' '.join(map(str, next_ai_12[6:]))}</h2></div>""", unsafe_allow_html=True)
        st.caption("※ 左上方第一個數字為機率最高推薦號碼")
    with a_c2:
        st.markdown("#### 📊 預測邏輯")
        st.write(f"1. **大數據機率**：分析最近 {window} 期號碼的頻率分佈與冷熱回歸趨勢。")
        st.write(f"2. **得分排序**：針對每個號碼計算「潛在開出機率分數」，分數最高者排列在前。")
        st.write(f"3. **歷史基準**：根據全期 {total_records} 筆數據進行動態加權。")
    st.markdown("</div>", unsafe_allow_html=True)

    # 歷史表現檢視 (100期)
    st.markdown("<h3 class='section-header'>📈 AI 歷史表現深度檢視 (最近 100 期分析)</h3>", unsafe_allow_html=True)
    backtest_data = get_historical_backtest(limit=100)
    
    # 圖表統計
    bt_df = pd.DataFrame(backtest_data)
    dist_data = bt_df['match_count'].value_counts().sort_index()
    
    tab_graph, tab_list = st.tabs(["📊 表現統計圖表", "📋 詳細期數清單 (3行格式)"])
    
    with tab_graph:
        g_c1, g_c2 = st.columns([1, 2])
        with g_c1:
            st.markdown("#### 🎯 命中次數分佈")
            fig_dist = px.bar(x=dist_data.index, y=dist_data.values, labels={'x': '命中字數', 'y': '期數'},
                              color=dist_data.values, color_continuous_scale='Oranges', template="plotly_dark")
            fig_dist.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig_dist, use_container_width=True)
        with g_c2:
            st.markdown("#### 📈 命中趨勢走勢")
            fig_trend = px.line(bt_df, x='date', y='match_count', labels={'date': '日期', 'match_count': '命中字數'},
                                template="plotly_dark", markers=True)
            fig_trend.update_traces(line_color='#FF6B35')
            fig_trend.update_layout(height=300)
            st.plotly_chart(fig_trend, use_container_width=True)

    with tab_list:
        st.markdown('<div style="height: 600px; overflow-y: scroll; padding: 15px; border: 1px solid rgba(255,255,255,0.1); border-radius: 10px; background: rgba(0,0,0,0.2);">', unsafe_allow_html=True)
        for res in backtest_data:
            badge_color = "#FF6B35" if res['match_count'] >= 3 else "#444"
            st.markdown(f"""
            <div class="history-card" style="border-left: 5px solid {badge_color};">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 1.1em; font-weight: bold;">📅 開獎日期：{res['date']}</span>
                    <span class="match-tag" style="background:{badge_color}; padding: 4px 12px;">中 {res['match_count']} 個字</span>
                </div>
                <div style="margin-top: 10px; padding: 5px; border-bottom: 1px solid rgba(255,255,255,0.05);">
                    <b style="color: #888;">第 1 行 (當期中獎數字)：</b> 
                    <span style="color:#7FD1B9; font-weight: bold; font-size: 1.1em;">{" , ".join(map(str, res['draw_nums']))}</span> 
                    <span style="color: #FF6B35;"> + 特別號: {res['extra']}</span>
                </div>
                <div style="margin-top: 5px; padding: 5px; border-bottom: 1px solid rgba(255,255,255,0.05);">
                    <b style="color: #888;">第 2 行 (AI 預測數字 - 按機率排序)：</b> 
                    {", ".join([f'<span style="color:{"#FF6B35" if n in res["matched_list"] else "#eee"}; font-weight:{"bold" if n in res["matched_list"] else "normal"}">{n}</span>' for n in res['prediction']])}
                </div>
                <div style="margin-top: 5px; padding: 5px; color: #FF6B35;">
                    <b>第 3 行 (分析結果)：</b> 本期 AI 成功預測命中 <span style="font-size: 1.2em; font-weight: bold;">{res['match_count']}</span> 個字。
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# --- 3. 歷史中獎檢查器 ---
if v_check:
    st.write("---")
    st.markdown("<h3 class='section-header'>🔎 歷史中獎檢查器 (49 號網格)</h3>", unsafe_allow_html=True)
    g_cols = st.columns(7)
    for i in range(1, 50):
        with g_cols[(i-1)%7]:
            is_s = i in st.session_state.selected_nums
            if st.button(f"{i:02d}", key=f"grid_{i}", width='stretch', type="primary" if is_s else "secondary"):
                if is_s: st.session_state.selected_nums.remove(i)
                elif len(st.session_state.selected_nums) < 6: st.session_state.selected_nums.add(i)
                st.rerun()
    sl = sorted(list(st.session_state.selected_nums))
    if len(sl) == 6:
        h_res = get_all_historical_wins_fast(tuple(sl), total_records)
        if h_res:
            st.success(f"🎉 歷史共中獎 {len(h_res)} 次")
            st.dataframe(pd.DataFrame(h_res).sort_values("Rank"), width='stretch', hide_index=True)
        else: st.info("此組合在歷史中未曾獲獎。")

# --- 4. 圖表分析 ---
if v_chart:
    st.write("---")
    t1, t2 = st.tabs(["📊 出字頻率", "📈 總和趨勢"])
    with t1:
        fq_df = pd.DataFrame(Counter(df_desc.head(window)[num_cols].values.flatten()).items(), columns=['No','Count']).sort_values('Count', ascending=False)
        st.plotly_chart(px.bar(fq_df.head(25), x='No', y='Count', color='Count', color_continuous_scale='Oranges', template="plotly_dark"), use_container_width=True)
    with t2:
        df_desc['draw_sum'] = df_desc[num_cols].sum(axis=1)
        st.plotly_chart(px.area(df_desc.head(window), x='date_parsed', y='draw_sum', template="plotly_dark", color_discrete_sequence=['#FF6B35']), use_container_width=True)
