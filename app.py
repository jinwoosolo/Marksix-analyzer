import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from collections import Counter
import datetime

# --- 基礎配置 ---
st.set_page_config(page_title="六合彩 AI 專業分析器 Pro", page_icon="🎰", layout="wide")

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
    # 確保數據按日期從舊到新排列以便回測
    return df.sort_values('date_parsed', ascending=True).reset_index(drop=True)

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
    # 我們將 df 載入為升序排列，方便歷史回測邏輯
    df_asc = load_data()
    # 用於顯示的則用降序排列
    df_desc = df_asc.sort_values('date_parsed', ascending=False).reset_index(drop=True)
    total_records = len(df_asc)
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
st.title("🎰 六合彩 AI 專業分析器 Pro")
latest = df_desc.iloc[0]
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
                for _, row in df_desc.iterrows():
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

# --- 2. 準確度回測實驗室 (模擬測試) ---
if show_backtest:
    st.write("---")
    st.markdown("<h3 class='section-header'>📈 AI 準確度回測實驗室 (模擬預測 18 個字)</h3>", unsafe_allow_html=True)
    st.write("系統會模擬 AI 策略（12熱 + 6冷 = 18個字），從歷史第一期起進行「逐期預測測試」。")
    
    # 用戶可以選擇回測深度
    test_options = ["最近 100 期", "最近 500 期", "全歷史紀錄 (需時較長)"]
    test_depth_sel = st.selectbox("選擇回測範圍", test_options, index=0)
    
    if test_depth_sel == "最近 100 期":
        start_idx = max(50, total_records - 100)
    elif test_depth_sel == "最近 500 期":
        start_idx = max(50, total_records - 500)
    else:
        start_idx = 50 # 從第 51 期開始預測 (前 50 期作基礎)

    if st.button("啟動全量回測分析", type="primary"):
        results_log = []
        prizes_count = Counter()
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_to_test = total_records - start_idx
        
        # 開始逐期循環
        for i in range(start_idx, total_records):
            # 目標期 (要預測的那一期)
            target_row = df_asc.iloc[i]
            # 參考期 (該期之前的 50 期數據)
            history_before = df_asc.iloc[i-50:i]
            
            # AI 18 字策略: 12熱 + 6冷
            flat_history = history_before[num_cols].values.flatten()
            freq = Counter(flat_history).most_common()
            hot_12 = [x[0] for x in freq[:12]]
            cold_6 = [x[0] for x in freq[-6:]]
            ai_18_pick = set(hot_12 + cold_6)
            
            # 獲取實際號碼以便對比
            actual_nums = [int(target_row[n]) for n in num_cols]
            special_num = int(target_row['extra'])
            
            # 檢查中獎情況
            prize_name, rank = calculate_prizes(ai_18_pick, actual_nums, special_num)
            
            if prize_name:
                prizes_count[prize_name] += 1
                results_log.append({
                    "日期": target_row['date'],
                    "AI 預測 (18字)": sorted(list(ai_18_pick)),
                    "當期開彩 (6+1)": f"{actual_nums} + ({special_num})",
                    "結果": prize_name
                })
            
            if (i - start_idx) % 50 == 0:
                progress_bar.progress((i - start_idx) / total_to_test)
                status_text.text(f"正在模擬分析日期: {target_row['date']}...")

        progress_bar.empty()
        status_text.empty()
        st.success(f"回測完成！共測試了 {total_to_test} 期。")
            
        # 顯示結果統計
        res_col1, res_col2 = st.columns([1, 2])
        with res_col1:
            st.metric("回測總期數", total_to_test)
            st.write("**獲獎匯總:**")
            if prizes_count:
                for p in ["1st Prize", "2nd Prize", "3rd Prize", "4th Prize", "5th Prize", "6th Prize", "7th Prize"]:
                    if prizes_count[p] > 0:
                        st.write(f"- **{p}**: {prizes_count[p]} 次")
            else:
                st.write("😔 未命中任何獎項（18字組合仍需極大運氣）。")
        
        with res_col2:
            if results_log:
                st.write("**詳細獲獎與對比紀錄:**")
                st.dataframe(pd.DataFrame(results_log), use_container_width=True)
            else:
                st.info("在回測範圍內，這套 18 字策略未能在模擬中獲獎。")

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

        # 搜尋歷史 (使用 desc 列表顯示最新結果優先)
        history_results = []
        for _, row in df_desc.iterrows():
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
    t1, t2, t3 = st.tabs(["📊 出字頻率", "📈 總和趨勢", "🧠 AI Oracle 預測 (18字大底)"])
    
    with t1:
        recent = df_desc.head(window_size)
        all_draws = recent[num_cols].values.flatten()
        freq_df = pd.DataFrame(Counter(all_draws).items(), columns=['Number', 'Count']).sort_values('Count', ascending=False)
        fig = px.bar(freq_df.head(25), x='Number', y='Count', color='Count', color_continuous_scale='Oranges', template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        df_desc['draw_sum'] = df_desc[num_cols].sum(axis=1)
        fig_trend = px.area(df_desc.head(window_size), x='date_parsed', y='draw_sum', template="plotly_dark", color_discrete_sequence=['#FF6B35'])
        st.plotly_chart(fig_trend, use_container_width=True)

    with t3:
        st.header("Prophet 智能預測引擎 (18字版本)")
        if st.button("生成下期預測報告", type="primary"):
            with st.spinner("正在計算 22 年數據與預測大底組合..."):
                p_df = df_desc.rename(columns={'date_parsed': 'ds', 'draw_sum': 'y'})[['ds', 'y']].dropna()
                m = Prophet(daily_seasonality=False, weekly_seasonality=True)
                m.fit(p_df)
                forecast = m.predict(m.make_future_dataframe(periods=1, freq='3D'))
                next_sum = forecast['yhat'].iloc[-1]
                
                c_low, c_high = st.columns([1, 2])
                with c_low:
                    fig_g = go.Figure(go.Indicator(mode="gauge+number", value=next_sum, title={'text': "預測總和指標"}, gauge={'bar':{'color':"#FF6B35"}}))
                    st.plotly_chart(fig_g, use_container_width=True)
                with c_high:
                    # 使用 12熱 + 6冷 = 18字策略
                    recent_counts = Counter(df_desc.head(window_size)[num_cols].values.flatten()).most_common()
                    hot = [int(x[0]) for x in recent_counts[:12]]
                    cold = [int(x[0]) for x in recent_counts[-6:]]
                    ai_18_pick = sorted(hot + cold)
                    st.markdown(f"""
                    <div style='background:rgba(255,107,53,0.1);padding:30px;border-radius:20px;text-align:center;border:2px solid #FF6B35;'>
                        <h3 style='color:white;'>AI 推薦 18 字大底組合</h3>
                        <h1 style='color:#FF6B35; letter-spacing:3px; font-size: 2.2em;'>
                            {' '.join(map(str, ai_18_pick[:9]))}<br>
                            {' '.join(map(str, ai_18_pick[9:]))}
                        </h1>
                        <p style='color:#aaa;'>策略：前 {window_size} 期最強 12 熱門字 + 6 絕冷門字</p>
                    </div>
                    """, unsafe_allow_html=True)
