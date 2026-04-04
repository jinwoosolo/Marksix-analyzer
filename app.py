import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from collections import Counter
import datetime
import numpy as np
import os
import itertools

# --- 1. APP 頁面配置 ---
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
    .missing-tag { background: #444; color: #aaa; padding: 2px 8px; border-radius: 4px; font-size: 0.9em; }
    .number-tag { display: inline-block; background: rgba(255, 107, 53, 0.1); border: 1px solid rgba(255, 107, 53, 0.2); padding: 2px 8px; margin: 2px; border-radius: 5px; color: #eee; font-family: monospace; }
    .excluded-tag { display: inline-block; background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.1); padding: 2px 8px; margin: 2px; border-radius: 5px; color: #666; font-family: monospace; text-decoration: line-through; }
    .comb-box { background: rgba(0, 0, 0, 0.3); padding: 15px; border-radius: 10px; font-family: 'Courier New', monospace; font-size: 0.85em; line-height: 1.8; height: 500px; overflow-y: scroll; border: 1px solid rgba(255,255,255,0.1); white-space: pre-wrap; }
    </style>
    """, unsafe_allow_html=True)

# --- 固定最愛組合 ---
FIXED_FAV_SETS = [
    {5, 11, 12, 13, 15, 27},
    {9, 10, 22, 24, 30, 49},
    {19, 23, 25, 39, 44, 46}
]

# --- 核心數據邏輯 ---
@st.cache_data(ttl=600)
def load_data():
    """載入數據並執行基本的清理，確保數值正確並防禦損壞數據"""
    if not os.path.exists('marksix.csv'):
        return pd.DataFrame(), np.array([]), np.array([]), np.array([])
        
    df = pd.read_csv('marksix.csv')
    df = df[df['date'].astype(str).str.len() <= 12] 
    num_cols_all = ['n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'extra']
    for col in num_cols_all:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=num_cols_all)
    df = df[df[num_cols_all].apply(lambda x: x.between(1, 49)).all(axis=1)]
    df['date_parsed'] = pd.to_datetime(df['date'], format='mixed')
    df = df.sort_values('date_parsed', ascending=True).reset_index(drop=True)
    draws_matrix = df[['n1', 'n2', 'n3', 'n4', 'n5', 'n6']].values
    extras_array = df['extra'].values
    dates_array = df['date'].values
    return df, draws_matrix, extras_array, dates_array

def calculate_ai_scores(historical_df, window_size=100, top_n=12):
    """根據頻率與遺漏值計算號碼綜合機率得分，並返回前 N 個號碼及其餘號碼"""
    recent_data = historical_df.tail(window_size)
    if recent_data.empty:
        full_list = list(range(1, 50))
        return full_list[:top_n], full_list[top_n:]
        
    nums = recent_data[['n1', 'n2', 'n3', 'n4', 'n5', 'n6']].values.flatten()
    counts = Counter(nums)
    scores = {}
    total_draws = len(recent_data)
    
    for n in range(1, 50):
        freq = counts.get(n, 0)
        freq_score = (freq / (total_draws * 6 / 49 + 0.001)) * 60 
        last_appearance = 0
        for i, draw in enumerate(reversed(recent_data[['n1', 'n2', 'n3', 'n4', 'n5', 'n6']].values)):
            if n in draw:
                last_appearance = i
                break
        gap_score = (last_appearance / 20) * 40
        scores[n] = freq_score + gap_score
        
    ranked_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_list = [int(x[0]) for x in ranked_nums[:top_n]]
    remaining_list = [int(x[0]) for x in ranked_nums[top_n:]]
    return top_list, remaining_list

@st.cache_data(ttl=600, show_spinner=False)
def get_historical_backtest(limit=100, top_n=12):
    """回測最近期的 AI 表現"""
    df_asc, draws_np, extras_np, dates_np = load_data()
    results = []
    if df_asc.empty: return []
    total = len(df_asc)
    start_idx = max(50, total - limit)
    
    for i in range(total - 1, start_idx - 1, -1):
        past_df = df_asc.iloc[:i]
        target_draw = set(draws_np[i])
        target_extra = int(extras_np[i])
        ai_top, ai_remain = calculate_ai_scores(past_df, window_size=100, top_n=top_n)
        draw_all = list(target_draw) + [target_extra]
        matched_nums = [n for n in ai_top if n in draw_all]
        match_count = len(matched_nums)
        
        results.append({
            "date": dates_np[i],
            "draw_nums": sorted(list(target_draw)),
            "extra": target_extra,
            "prediction": ai_top,
            "missing": sorted(ai_remain),
            "match_count": match_count,
            "matched_list": matched_nums
        })
    return results

@st.cache_data(show_spinner=False)
def get_all_historical_wins_fast(user_set_tuple, total_count):
    df, draws, extras, dates = load_data()
    if df.empty: return []
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
    if not df_asc.empty:
        df_desc = df_asc.sort_values('date_parsed', ascending=False).reset_index(drop=True)
        total_records = len(df_asc)
        num_cols = ['n1', 'n2', 'n3', 'n4', 'n5', 'n6']
        latest = df_desc.iloc[0]
    else:
        st.warning("數據庫連線中或 marksix.csv 缺失。")
        st.stop()
except Exception as e:
    st.error(f"數據載入出錯: {e}"); st.stop()

if 'selected_nums' not in st.session_state: st.session_state.selected_nums = set()

# --- 側邊欄 ---
with st.sidebar:
    st.title("⚙️ 控制面板")
    v_fav = st.checkbox("顯示組合追蹤", value=True)
    v_ai = st.checkbox("顯示 AI 預測與回測", value=True)
    v_check = st.checkbox("顯示中獎檢查器", value=True)
    v_chart = st.checkbox("顯示分析圖表", value=True)
    st.divider()
    window = st.slider("預測參考窗口", 50, 500, 100)
    if st.button("🔄 手動刷新緩存"):
        st.cache_data.clear()
        st.rerun()

# --- 頁首資訊 ---
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
    
    next_ai_12, _ = calculate_ai_scores(df_asc, window_size=window, top_n=12)
    next_ai_44, next_excluded_5 = calculate_ai_scores(df_asc, window_size=window, top_n=44)
    next_remaining_32 = [n for n in next_ai_44 if n not in next_ai_12]
    
    a_tab1, a_tab2, a_tab3, a_tab4 = st.tabs(["🎯 12 字重點推薦", "廣 44 字高覆蓋推薦", "📊 下期預測完整清單", "🧩 組合拆解分析"])
    
    with a_tab1:
        st.markdown("<div class='ai-box'>", unsafe_allow_html=True)
        st.markdown("#### 🎯 下期推薦 12 字 (按機率排序)")
        st.markdown(f"""<div style='background: linear-gradient(45deg, #FF6B35, #F7931E); padding: 25px; border-radius: 15px; text-align: center; color: white;'><h2 style='letter-spacing: 3px;'>{' '.join(map(str, next_ai_12[:6]))}<br>{' '.join(map(str, next_ai_12[6:]))}</h2></div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with a_tab2:
        st.markdown("<div class='ai-box'>", unsafe_allow_html=True)
        st.markdown("#### 🎯 下期推薦 44 字 (高覆蓋候選池)")
        st.write(", ".join(map(str, next_ai_44)))
        st.markdown("---")
        st.markdown("#### ❌ AI 本期排除號碼 (5 字)")
        st.write(", ".join(map(str, sorted(next_excluded_5))))
        st.markdown("</div>", unsafe_allow_html=True)

    with a_tab3:
        st.markdown("<div class='ai-box'>", unsafe_allow_html=True)
        st.markdown("#### 📊 下期預測全數據分類")
        st.markdown("##### 1. 核心推薦 (12個字)")
        st.markdown(" ".join([f"<span class='number-tag' style='border-color:#FF6B35; color:#FF6B35;'>{n}</span>" for n in next_ai_12]), unsafe_allow_html=True)
        st.markdown("##### 2. 次選及後備 (32個字)")
        st.markdown(" ".join([f"<span class='number-tag'>{n}</span>" for n in next_remaining_32]), unsafe_allow_html=True)
        st.markdown("##### 3. AI 排除/極低機會 (5個字)")
        st.markdown(" ".join([f"<span class='excluded-tag'>{n}</span>" for n in sorted(next_excluded_5)]), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with a_tab4:
        st.markdown("<div class='ai-box'>", unsafe_allow_html=True)
        st.markdown("#### 🧩 推薦號碼自動組合拆解 (完整清單)")
        
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("##### 核心 12 字 → 2 字一組 (共 66 組)")
            comb_12_2 = list(itertools.combinations(sorted(next_ai_12), 2))
            comb_text_12 = ""
            for i, combo in enumerate(comb_12_2):
                comb_text_12 += f"[{i+1:02d}] {combo[0]:02d}, {combo[1]:02d}    "
                if (i+1) % 2 == 0: comb_text_12 += "\n"
            st.markdown(f"<div class='comb-box'>{comb_text_12}</div>", unsafe_allow_html=True)
            
        with c2:
            st.markdown("##### 次選 32 字 → 3 字一組 (共 4,960 組)")
            comb_32_3 = list(itertools.combinations(sorted(next_remaining_32), 3))
            
            # 提供下載
            df_comb_32 = pd.DataFrame(comb_32_3, columns=['碼1', '碼2', '碼3'])
            csv_comb = df_comb_32.to_csv(index=False).encode('utf-8-sig')
            st.download_button(label="📥 下載完整 4,960 組組合 (CSV)", data=csv_comb, file_name=f"marksix_32_select_3_{datetime.date.today()}.csv", mime="text/csv")
            
            # 列出所有組合
            comb_text_32 = ""
            for i, combo in enumerate(comb_32_3):
                comb_text_32 += f"{combo[0]:02d},{combo[1]:02d},{combo[2]:02d} | "
                if (i+1) % 4 == 0: comb_text_32 += "\n"
            
            st.markdown(f"<div class='comb-box'>{comb_text_32}</div>", unsafe_allow_html=True)
            
        st.markdown("</div>", unsafe_allow_html=True)

    # 歷史表現檢視
    st.markdown("<h3 class='section-header'>📈 AI 歷史表現深度檢視</h3>", unsafe_allow_html=True)
    tab_graph, tab_list_12, tab_list_44 = st.tabs(["📊 表現統計圖表", "📋 12 字回測清單", "📋 44 字高覆蓋回測"])
    
    backtest_12 = get_historical_backtest(limit=100, top_n=12)
    backtest_44 = get_historical_backtest(limit=100, top_n=44)
    
    with tab_graph:
        if backtest_12:
            bt_df = pd.DataFrame(backtest_12)
            dist_data = bt_df['match_count'].value_counts().sort_index()
            g_c1, g_c2 = st.columns([1, 2])
            with g_c1:
                st.plotly_chart(px.bar(x=dist_data.index, y=dist_data.values, color_continuous_scale='Oranges', template="plotly_dark"), use_container_width=True)
            with g_c2:
                st.plotly_chart(px.line(bt_df, x='date', y='match_count', template="plotly_dark", markers=True), use_container_width=True)

    with tab_list_12:
        st.markdown('<div style="height: 600px; overflow-y: scroll; padding: 15px; border-radius: 10px; background: rgba(0,0,0,0.2);">', unsafe_allow_html=True)
        for res in backtest_12:
            badge_color = "#FF6B35" if res['match_count'] >= 3 else "#444"
            st.markdown(f"""<div class="history-card" style="border-left: 5px solid {badge_color};"><b>📅 日期：{res['date']}</b> <span class="match-tag" style="background:{badge_color}">中 {res['match_count']} 字</span><br>結果：<span style="color:#7FD1B9">{" , ".join(map(str, res['draw_nums']))}</span> + ({res['extra']})<br>AI：{", ".join([f'<span style="color:{"#FF6B35" if n in res["matched_list"] else "#eee"}">{n}</span>' for n in res['prediction']])}</div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_list_44:
        st.markdown('<div style="height: 600px; overflow-y: scroll; padding: 15px; border-radius: 10px; background: rgba(0,0,0,0.2);">', unsafe_allow_html=True)
        for res in backtest_44:
            badge_color = "#7FD1B9" if res['match_count'] >= 7 else "#FF6B35" if res['match_count'] >= 6 else "#444"
            st.markdown(f"""<div class="history-card" style="border-left: 5px solid {badge_color};"><b>📅 日期：{res['date']}</b> <span class="match-tag" style="background:{badge_color}">中 {res['match_count']} 字</span><br>第1行：<small style="color:#bbb;">{", ".join(map(str, res['prediction']))}</small><br>第2行：<small class="missing-tag">{" , ".join(map(str, res['missing']))}</small><br>第3行：<span style="color:#7FD1B9">{" , ".join(map(str, res['draw_nums']))}</span> + ({res['extra']})</div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# --- 3. 歷史中獎檢查器 ---
if v_check:
    st.write("---")
    st.markdown("<h3 class='section-header'>🔎 歷史中獎檢查器 (49 號網格)</h3>", unsafe_allow_html=True)
    g_cols = st.columns(7)
    for i in range(1, 50):
        with g_cols[(i-1)%7]:
            is_s = i in st.session_state.selected_nums
            if st.button(f"{i:02d}", key=f"grid_{i}", use_container_width=True, type="primary" if is_s else "secondary"):
                if is_s: st.session_state.selected_nums.remove(i)
                elif len(st.session_state.selected_nums) < 6: st.session_state.selected_nums.add(i)
                st.rerun()
    sl = sorted(list(st.session_state.selected_nums))
    if len(sl) == 6:
        h_res = get_all_historical_wins_fast(tuple(sl), total_records)
        if h_res:
            st.success(f"🎉 歷史共中獎 {len(h_res)} 次")
            st.dataframe(pd.DataFrame(h_res).sort_values("Rank"), use_container_width=True, hide_index=True)
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
