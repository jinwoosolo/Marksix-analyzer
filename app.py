import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from collections import Counter
import datetime

# --- 基礎配置 ---
st.set_page_config(page_title="六合彩 AI 專業分析器", page_icon="🎰", layout="wide")

# 專業介面 CSS
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
    .section-header { margin-top: 0px; margin-bottom: 10px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- 持久化邏輯 (網址參數) ---
def sync_favs_to_url():
    params = {}
    for i, fav in enumerate(st.session_state.fav_sets):
        if fav:
            params[f"fav{i}"] = ",".join(map(str, sorted(list(fav))))
    st.query_params.update(params)

def load_favs_from_url():
    favs = [None, None, None]
    for i in range(3):
        param_val = st.query_params.get(f"fav{i}")
        if param_val:
            try:
                favs[i] = [int(x) for x in param_val.split(",")]
            except: pass
    return favs

# --- 初始化狀態 ---
if 'selected_nums' not in st.session_state:
    st.session_state.selected_nums = set()
if 'fav_sets' not in st.session_state:
    st.session_state.fav_sets = load_favs_from_url()

@st.cache_data(ttl=3600)
def load_data():
    df = pd.read_csv('marksix.csv')
    df['date_parsed'] = pd.to_datetime(df['date'], format='mixed')
    return df.sort_values('date_parsed', ascending=False).reset_index(drop=True)

def calculate_prizes(user_set, draw_set, extra_num):
    """計算單次開獎的中獎等級"""
    matched = set(user_set).intersection(set(draw_set))
    match_count = len(matched)
    has_extra = extra_num in user_set
    
    if match_count == 6: return "1st Prize", 1
    if match_count == 5 and has_extra: return "2nd Prize", 2
    if match_count == 5: return "3rd Prize", 3
    if match_count == 4 and has_extra: return "4th Prize", 4
    if match_count == 4: return "5th Prize", 5
    if match_count == 3 and has_extra: return "6th Prize", 6
    if match_count == 3: return "7th Prize", 7
    return None, 99

try:
    df = load_data()
    total_records = len(df)
    num_cols = ['n1', 'n2', 'n3', 'n4', 'n5', 'n6']
except Exception as e:
    st.error(f"⚠️ 載入數據出錯: {e}")
    st.stop()

# --- 側邊欄控制 ---
with st.sidebar:
    st.title("控制面板")
    st.subheader("顯示設定")
    show_favs = st.checkbox("顯示收藏追蹤", value=True)
    show_checker = st.checkbox("顯示中獎檢查器", value=True)
    show_analysis = st.checkbox("顯示分析圖表", value=True)
    show_backtest = st.checkbox("顯示 AI 準確度回測實驗室", value=False)
    st.divider()
    if show_analysis:
        window_size = st.slider("統計窗口 (期數)", 10, total_records, 100)
    st.info(f"📊 總開彩次數: {total_records}")

# --- 頁首資訊 ---
st.title("🎰 六合彩 AI 專業分析器")
latest = df.iloc[0]
l_nums = "  ".join([str(int(latest[n])) for n in num_cols])

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f"<div class='metric-card'><span class='stat-label'>最新開彩日期</span><br><span class='stat-val'>{latest['date']}</span></div>", unsafe_allow_html=True)
with c2:
    st.markdown(f"<div class='metric-card'><span class='stat-label'>中獎號碼</span><br><span class='stat-val'>{l_nums}</span></div>", unsafe_allow_html=True)
with c3:
    st.markdown(f"<div class='metric-card'><span class='stat-label'>特別號碼</span><br><span class='stat-val' style='color:#7FD1B9'>{int(latest['extra'])}</span></div>", unsafe_allow_html=True)

# --- 1. 收藏追蹤 ---
if show_favs:
    st.markdown("<h3 class='section-header'>⭐ 我的最愛組合追蹤</h3>", unsafe_allow_html=True)
    fav_cols = st.columns(3)
    for idx, fav in enumerate(st.session_state.fav_sets):
        with fav_cols[idx]:
            if fav:
                st.markdown(f"<div class='favorite-card'>", unsafe_allow_html=True)
                st.markdown(f"**位置 {idx+1}**")
                st.code(f"{' '.join(map(str, fav))}")
                
                # 計算歷史表現
                all_results = []
                for _, row in df.iterrows():
                    p, r = calculate_prizes(fav, [row[n] for n in num_cols], row['extra'])
                    if p: all_results.append({"Date": row['date'], "Prize": p, "Rank": r})
                
                high_tier = [r for r in all_results if r['Rank'] <= 4]
                latest_draw_set = {latest[n] for n in num_cols}
                latest_match = len(set(fav).intersection(latest_draw_set))
                
                st.markdown(f"歷史中獎總數: **{len(all_results)}**")
                st.markdown(f"大獎次數 (4獎+): <span class='win-highlight'>{len(high_tier)}</span>", unsafe_allow_html=True)
                
                if high_tier:
                    dates = [r['Date'] for r in high_tier]
                    st.markdown(f"<div class='date-list' style='font-size:0.75em;'><b>大獎日期:</b> {', '.join(dates)}</div>", unsafe_allow_html=True)
                
                if latest_match >= 3:
                    st.warning(f"🔔 最新一期中 {latest_match} 個字!")
                
                if st.button(f"清除位置 {idx+1}", key=f"clear_{idx}", use_container_width=True):
                    st.session_state.fav_sets[idx] = None
                    sync_favs_to_url(); st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.info(f"位置 {idx+1} 尚未儲存")

# --- 2. 準確度回測實驗室 (新增) ---
if show_backtest:
    st.write("---")
    st.markdown("<h3 class='section-header'>📈 AI 準確度回測實驗室 (模擬測試)</h3>", unsafe_allow_html=True)
    st.write("此功能會「假裝不知道過去結果」，模擬 AI 在過去每一期選出的號碼，並計算實際回報。")
    
    test_range = st.select_slider("回測範圍 (最近幾多期)", options=[50, 100, 200, 500], value=100)
    
    if st.button("開始大規模回測分析", type="primary"):
        results_log = []
        prizes_count = Counter()
        
        with st.spinner(f"正在分析過去 {test_range} 期的預測表現..."):
            # 我們從較舊的日期開始往最新日期推算
            test_df = df.head(test_range + 50) # 多取一點數據作為初始統計窗口
            
            for i in range(test_range, 0, -1):
                # 目前要「預測」的那一期
                target_row = df.iloc[i-1]
                # 預測時只能參考 target 之前的數據
                history_before = df.iloc[i:i+50] 
                
                if len(history_before) < 50: continue
                
                # AI 策略: 4熱 + 2冷 (基於當時的 50 期窗口)
                flat_history = history_before[num_cols].values.flatten()
                freq = Counter(flat_history).most_common()
                hot = [x[0] for x in freq[:4]]
                cold = [x[0] for x in freq[-2:]]
                ai_pick = set(hot + cold)
                
                # 檢查結果
                draw_nums = [target_row[n] for n in num_cols]
                prize_name, rank = calculate_prizes(ai_pick, draw_nums, target_row['extra'])
                
                if prize_name:
                    prizes_count[prize_name] += 1
                    results_log.append({"日期": target_row['date'], "AI 號碼": sorted(list(ai_pick)), "結果": prize_name})
            
        # 顯示結果
        res_col1, res_col2 = st.columns([1, 2])
        with res_col1:
            st.metric("總測試期數", test_range)
            st.write("**獲獎統計:**")
            if prizes_count:
                for p, c in prizes_count.items():
                    st.write(f"- {p}: {c} 次")
            else:
                st.write("😔 在這段期間內未命中任何獎項。")
        
        with res_col2:
            if results_log:
                st.write("**詳細獲獎紀錄:**")
                st.table(pd.DataFrame(results_log))
            else:
                st.info("統計學提示：頭獎機率極低。此 AI 策略旨在提高命中 7 獎的頻率。")

# --- 3. 中獎檢查器 ---
if show_checker:
    st.write("---")
    h_col1, h_col2 = st.columns([5, 1])
    with h_col1: st.markdown("<h3 class='section-header'>🔎 歷史中獎檢查器</h3>", unsafe_allow_html=True)
    with h_col2:
        if st.button("重置選擇", use_container_width=True):
            st.session_state.selected_nums = set(); st.rerun()

    st.write("點擊下方號碼選擇 6 個字：")
    grid = st.columns(7)
    for i in range(1, 50):
        with grid[(i-1) % 7]:
            sel = i in st.session_state.selected_nums
            if st.button(f"{i:02d}", key=f"grid_{i}", use_container_width=True, type="primary" if sel else "secondary"):
                if sel: st.session_state.selected_nums.remove(i)
                elif len(st.session_state.selected_nums) < 6: st.session_state.selected_nums.add(i)
                st.rerun()

    selected_list = sorted(list(st.session_state.selected_nums))
    st.markdown(f"**已選號碼:** `{', '.join(map(str, selected_list)) if selected_list else '未選擇'}` ({len(selected_list)}/6)")

    if len(selected_list) == 6:
        st.write("💾 **儲存到我的最愛:**")
        scols = st.columns(3)
        for idx in range(3):
            with scols[idx]:
                if st.button(f"儲存至位置 {idx+1}", key=f"save_{idx}", use_container_width=True):
                    st.session_state.fav_sets[idx] = selected_list
                    sync_favs_to_url(); st.toast("已儲存！"); st.rerun()

        # 搜尋歷史
        history_results = []
        for _, row in df.iterrows():
            p, r = calculate_prizes(selected_list, [row[n] for n in num_cols], row['extra'])
            if p: history_results.append({"Date": row['date'], "Prize": p, "Rank": r})
        
        if history_results:
            st.success(f"🎉 歷史上共中獎 **{len(history_results)}** 次")
            res_df = pd.DataFrame(history_results).sort_values("Rank")
            st.dataframe(res_df[["Date", "Prize"]], use_container_width=True, hide_index=True)
        else:
            st.info("此組合在歷史紀錄中未曾中過獎。")

# --- 4. 分析圖表 ---
if show_analysis:
    st.write("---")
    t1, t2, t3 = st.tabs(["📊 出字頻率", "📈 總和趨勢", "🧠 AI Oracle 預測"])
    
    with t1:
        recent = df.head(window_size)
        all_draws = recent[num_cols].values.flatten()
        freq_df = pd.DataFrame(Counter(all_draws).items(), columns=['Number', 'Count']).sort_values('Count', ascending=False)
        fig = px.bar(freq_df.head(25), x='Number', y='Count', color='Count', color_continuous_scale='Oranges', template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        df['draw_sum'] = df[num_cols].sum(axis=1)
        fig_trend = px.area(df.head(window_size), x='date_parsed', y='draw_sum', template="plotly_dark", color_discrete_sequence=['#FF6B35'])
        st.plotly_chart(fig_trend, use_container_width=True)

    with t3:
        st.header("Prophet 智能預測引擎")
        if st.button("生成下期預測報告", type="primary"):
            with st.spinner("正在計算統計模型..."):
                p_df = df.rename(columns={'date_parsed': 'ds', 'draw_sum': 'y'})[['ds', 'y']].dropna()
                m = Prophet(daily_seasonality=False, weekly_seasonality=True)
                m.fit(p_df)
                forecast = m.predict(m.make_future_dataframe(periods=1, freq='3D'))
                next_sum = forecast['yhat'].iloc[-1]
                
                c_low, c_high = st.columns(2)
                with c_low:
                    fig_g = go.Figure(go.Indicator(mode="gauge+number", value=next_sum, title={'text': "預測總和指標"}, gauge={'bar':{'color':"#FF6B35"}}))
                    st.plotly_chart(fig_g, use_container_width=True)
                with c_high:
                    # 使用熱冷邏輯
                    recent_counts = Counter(df.head(50)[num_cols].values.flatten()).most_common()
                    hot = [int(x[0]) for x in recent_counts[:4]]
                    cold = [int(x[0]) for x in recent_counts[-2:]]
                    ai_pick = sorted(hot + cold)
                    st.markdown(f"<div style='background:rgba(255,107,53,0.1);padding:40px;border-radius:20px;text-align:center;border:2px solid #FF6B35;'><h3 style='color:white;'>AI 推薦組合</h3><h1 style='color:#FF6B35;letter-spacing:8px;'>{' '.join(map(str, ai_pick))}</h1><p>基於「4熱 2冷」策略</p></div>", unsafe_allow_html=True)
