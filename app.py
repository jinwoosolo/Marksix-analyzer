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
    .date-list { font-size: 0.75em; color: #aaa; margin-top: 5px; line-height: 1.5; }
    .section-header { margin-top: 15px; margin-bottom: 15px; font-weight: bold; border-left: 5px solid #FF6B35; padding-left: 10px; }
    .ai-analysis-box { background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 107, 53, 0.2); border-radius: 12px; padding: 15px; color: #e0e0e0; line-height: 1.6; }
    </style>
    """, unsafe_allow_html=True)

# --- 固定最愛組合 ---
# 設定您要求的三組固定號碼
FIXED_FAV_SETS = [
    {5, 11, 12, 13, 15, 27},
    {9, 10, 22, 24, 30, 49},
    {19, 23, 25, 39, 44, 46}
]

# --- 核心數據邏輯 ---
@st.cache_data(ttl=3600)
def load_data():
    """載入數據並預處理為 NumPy 格式以提升速度"""
    df = pd.read_csv('marksix.csv')
    df['date_parsed'] = pd.to_datetime(df['date'], format='mixed')
    df = df.sort_values('date_parsed', ascending=True).reset_index(drop=True)
    
    # 預提取 NumPy 數組以供高速運算
    draws_matrix = df[['n1', 'n2', 'n3', 'n4', 'n5', 'n6']].values
    extras_array = df['extra'].values
    dates_array = df['date'].values
    
    return df, draws_matrix, extras_array, dates_array

@st.cache_data(show_spinner=False)
def get_all_historical_wins_fast(user_set_tuple, total_count):
    """NumPy 優化版歷史中獎檢查器"""
    _, draws, extras, dates = load_data()
    user_set = set(user_set_tuple)
    results = []
    
    for i in range(len(draws)):
        matched = 0
        for num in draws[i]:
            if num in user_set:
                matched += 1
        
        if matched < 3:
            continue
            
        has_extra = int(extras[i]) in user_set
        
        prize, rank = None, 99
        if matched == 6: prize, rank = "1st Prize", 1
        elif matched == 5 and has_extra: prize, rank = "2nd Prize", 2
        elif matched == 5: prize, rank = "3rd Prize", 3
        elif matched == 4 and has_extra: prize, rank = "4th Prize", 4
        elif matched == 4: prize, rank = "5th Prize", 5
        elif matched == 3 and has_extra: prize, rank = "6th Prize", 6
        elif matched == 3: prize, rank = "7th Prize", 7
        
        if prize:
            results.append({"Date": dates[i], "Prize": prize, "Rank": rank})
            
    # 按獎項等級排序（1獎優先），同等級按日期排序（最新優先）
    return sorted(results, key=lambda x: (x['Rank'], x['Date']), reverse=False)

try:
    df_asc, draws_np, extras_np, dates_np = load_data()
    df_desc = df_asc.sort_values('date_parsed', ascending=False).reset_index(drop=True)
    total_records = len(df_asc)
    num_cols = ['n1', 'n2', 'n3', 'n4', 'n5', 'n6']
except Exception as e:
    st.error(f"數據載入出錯: {e}"); st.stop()

# --- 初始化狀態 ---
if 'selected_nums' not in st.session_state: 
    st.session_state.selected_nums = set()

# --- 側邊欄 ---
with st.sidebar:
    st.title("⚙️ 控制面板")
    st.subheader("顯示設定")
    v_fav = st.checkbox("顯示組合追蹤", value=True)
    v_ai = st.checkbox("顯示 AI 預測專區", value=True)
    v_check = st.checkbox("顯示中獎檢查器", value=True)
    v_chart = st.checkbox("顯示分析圖表", value=True)
    v_test = st.checkbox("顯示回測實驗室", value=False)
    st.divider()
    window = st.slider("統計窗口 (期數)", 10, 500, 100)
    st.info(f"📊 總開彩: {total_records} 期")

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
            
            # 獲取歷史獲獎數據
            f_res = get_all_historical_wins_fast(tuple(sorted_fav), total_records)
            
            # 大獎定義為 4 獎或以上 (Rank 1-4)
            high = [r for r in f_res if r['Rank'] <= 4]
            latest_nums_set = {int(latest[n]) for n in num_cols}
            l_match = len(fav.intersection(latest_nums_set))
            
            st.markdown(f"總中獎次數: **{len(f_res)}** | 大獎: <span class='win-highlight'>{len(high)}</span>", unsafe_allow_html=True)
            
            if high:
                # 格式化顯示近期大獎的日期與獎項
                formatted_wins = [f"{r['Date']} ({r['Prize']})" for r in high[:5]]
                wins_html = "<br>".join(formatted_wins)
                st.markdown(f"<div class='date-list'><b>近期大獎紀錄:</b><br>{wins_html}{'<br>...' if len(high) > 5 else ''}</div>", unsafe_allow_html=True)
            
            if l_match >= 3: 
                st.warning(f"🔔 最新一期中 {l_match} 字！")
            else:
                st.write(f"最新一期中 {l_match} 字")
            
            st.markdown("</div>", unsafe_allow_html=True)

# --- 2. AI 智能預測 ---
if v_ai:
    st.write("---")
    st.markdown("<h3 class='section-header'>🔮 AI 智能預測與深度分析 (下期推薦 12 字)</h3>", unsafe_allow_html=True)
    if st.button("🚀 生成下期 AI 智能深度分析報告", type="primary", width='stretch'):
        with st.spinner("分析中..."):
            df_desc['draw_sum'] = df_desc[num_cols].sum(axis=1)
            p_df = df_desc.rename(columns={'date_parsed': 'ds', 'draw_sum': 'y'})[['ds', 'y']].dropna()
            m = Prophet(daily_seasonality=False, weekly_seasonality=True)
            m.fit(p_df)
            forecast = m.predict(m.make_future_dataframe(periods=1, freq='3D'))
            n_sum = forecast['yhat'].iloc[-1]
            f_counts = Counter(df_desc.head(window)[num_cols].values.flatten()).most_common()
            h_8 = [int(x[0]) for x in f_counts[:8]]
            c_4 = [int(x[0]) for x in f_counts[-4:]]
            ai_12 = sorted(h_8 + c_4)
            a_c1, a_c2 = st.columns([1, 1.5])
            with a_c1:
                st.markdown("#### 🎯 AI 推薦 12 字組合")
                st.markdown(f"<div style='background: linear-gradient(45deg, #FF6B35, #F7931E); padding: 25px; border-radius: 15px; text-align: center; color: white;'><h1 style='font-size: 2.2em; letter-spacing: 5px;'>{' '.join(map(str, ai_12[:6]))}<br>{' '.join(map(str, ai_12[6:]))}</h1></div>", unsafe_allow_html=True)
                fig_g = go.Figure(go.Indicator(mode="gauge+number", value=n_sum, title={'text': "下期總和預測"}, gauge={'bar':{'color':"#FF6B35"}, 'axis':{'range':[21,279]}}))
                fig_g.update_layout(height=230, margin=dict(l=10, r=10, t=40, b=0), paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_g, width='stretch')
            with a_c2:
                st.markdown("#### 📝 預測邏輯分析")
                st.markdown(f"""<div class="ai-analysis-box"><strong>1. 總和趨勢：</strong> Prophet 模型預測總和約 <strong>{n_sum:.1f}</strong>。代表偏向於 <strong>{"大數組合" if n_sum > 150 else "小數組合"}</strong>。<br><br><strong>2. 策略：</strong> 結合 8 熱門號碼與 4 個冷門回歸號碼。<br><br><strong>3. 平衡：</strong> {len([x for x in ai_12 if x%2!=0])}單:{len([x for x in ai_12 if x%2==0])}雙 | {len([x for x in ai_12 if x>24])}大:{len([x for x in ai_12 if x<=24])}小</div>""", unsafe_allow_html=True)

# --- 3. 歷史中獎檢查器 ---
if v_check:
    st.write("---")
    ch_c1, ch_c2 = st.columns([5, 1])
    with ch_c1: st.markdown("<h3 class='section-header'>🔎 歷史中獎檢查器 (49 號網格)</h3>", unsafe_allow_html=True)
    with ch_c2: 
        if st.button("重置選擇", width='stretch'): st.session_state.selected_nums = set(); st.rerun()
    
    g_cols = st.columns(7)
    for i in range(1, 50):
        with g_cols[(i-1)%7]:
            is_s = i in st.session_state.selected_nums
            if st.button(f"{i:02d}", key=f"grid_{i}", width='stretch', type="primary" if is_s else "secondary"):
                if is_s: st.session_state.selected_nums.remove(i)
                elif len(st.session_state.selected_nums) < 6: st.session_state.selected_nums.add(i)
                st.rerun()
                
    sl = sorted(list(st.session_state.selected_nums))
    st.write(f"**已選組合:** `{sl if sl else '尚未選擇'}` ({len(sl)}/6)")
    
    if len(sl) == 6:
        h_res = get_all_historical_wins_fast(tuple(sl), total_records)
        if h_res:
            st.success(f"🎉 歷史共中獎 {len(h_res)} 次")
            st.dataframe(pd.DataFrame(h_res).sort_values("Rank"), width='stretch', hide_index=True)
        else:
            st.info("此組合在歷史中未曾獲獎。")

# --- 4. 回測實驗室 ---
if v_test:
    st.write("---")
    st.markdown("<h3 class='section-header'>📈 AI 準確度回測實驗室 (12字策略)</h3>", unsafe_allow_html=True)
    scope = st.selectbox("選擇回測範圍", ["最近 100 期", "最近 500 期", "全歷史紀錄"])
    s_idx = 50 if "全" in scope else (total_records-100 if "100" in scope else total_records-500)
    if st.button("執行 12 字策略回測", width='stretch'):
        log, pc = [], Counter()
        prog = st.progress(0)
        for i in range(max(50, s_idx), total_records):
            hist_draws = draws_np[i-50:i].flatten()
            fq = Counter(hist_draws).most_common()
            ai_12_back = {int(x[0]) for x in fq[:8]} | {int(x[0]) for x in fq[-4:]}
            target_draw = set(draws_np[i])
            matched = len(ai_12_back.intersection(target_draw))
            has_e = int(extras_np[i]) in ai_12_back
            p_name = None
            if matched == 6: p_name = "1st Prize"
            elif matched == 5 and has_e: p_name = "2nd Prize"
            elif matched == 5: p_name = "3rd Prize"
            elif matched == 4 and has_e: p_name = "4th Prize"
            elif matched == 4: p_name = "5th Prize"
            elif matched == 3 and has_e: p_name = "6th Prize"
            elif matched == 3: p_name = "7th Prize"
            if p_name:
                pc[p_name] += 1
                log.append({"日期": dates_np[i], "結果": p_name})
            prog.progress((i-s_idx)/(total_records-s_idx))
        st.write(pc); st.dataframe(pd.DataFrame(log), width='stretch')

# --- 5. 圖表分析 ---
if v_chart:
    st.write("---")
    t1, t2 = st.tabs(["📊 出字頻率", "📈 總和趨勢"])
    with t1:
        fq_df = pd.DataFrame(Counter(df_desc.head(window)[num_cols].values.flatten()).items(), columns=['No','Count']).sort_values('Count', ascending=False)
        st.plotly_chart(px.bar(fq_df.head(25), x='No', y='Count', color='Count', color_continuous_scale='Oranges', template="plotly_dark"), width='stretch')
    with t2:
        df_desc['draw_sum'] = df_desc[num_cols].sum(axis=1)
        st.plotly_chart(px.area(df_desc.head(window), x='date_parsed', y='draw_sum', template="plotly_dark", color_discrete_sequence=['#FF6B35']), width='stretch')
