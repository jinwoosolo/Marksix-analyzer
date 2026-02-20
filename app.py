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
    .section-header { margin-top: 0px; margin-bottom: 15px; font-weight: bold; border-left: 5px solid #FF6B35; padding-left: 10px; }
    .ai-analysis-box {
        background: rgba(255, 107, 53, 0.05);
        border: 1px solid rgba(255, 107, 53, 0.2);
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 20px;
    }
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
    return df.sort_values('date_parsed', ascending=True).reset_index(drop=True)

def calculate_prizes(user_set, draw_set, extra_num):
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
    df_asc = load_data()
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
    show_ai_predict = st.checkbox("顯示 AI 智能預測專區", value=True)
    show_checker = st.checkbox("顯示中獎檢查器", value=True)
    show_analysis = st.checkbox("顯示分析圖表", value=True)
    show_backtest = st.checkbox("顯示 AI 準確度回測實驗室", value=False)
    st.divider()
    if show_analysis or show_ai_predict:
        window_size = st.slider("統計窗口 (期數)", 10, 500, 100)
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
                if latest_match >= 3: st.warning(f"🔔 最新一期中 {latest_match} 個字!")
                if st.button(f"清除位置 {idx+1}", key=f"clear_{idx}", use_container_width=True):
                    st.session_state.fav_sets[idx] = None
                    sync_favs_to_url(); st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.info(f"位置 {idx+1} 尚未儲存")

# --- 2. AI 智能預測專區 (新 Section) ---
if show_ai_predict:
    st.write("---")
    st.markdown("<h3 class='section-header'>🔮 AI 智能預測分析專區 (預計下期 12 字)</h3>", unsafe_allow_html=True)
    
    if st.button("🚀 執行下期 AI 智能深度分析", type="primary", use_container_width=True):
        with st.spinner("正在融合 Prophet 趨勢模型與熱冷統計學..."):
            # 數據準備
            df_desc['draw_sum'] = df_desc[num_cols].sum(axis=1)
            p_df = df_desc.rename(columns={'date_parsed': 'ds', 'draw_sum': 'y'})[['ds', 'y']].dropna()
            
            # 1. Prophet 預測總和
            m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
            m.fit(p_df)
            forecast = m.predict(m.make_future_dataframe(periods=1, freq='3D'))
            next_sum = forecast['yhat'].iloc[-1]
            next_sum_lower = forecast['yhat_lower'].iloc[-1]
            next_sum_upper = forecast['yhat_upper'].iloc[-1]
            
            # 2. 號碼選擇邏輯 (8熱 + 4冷)
            recent_counts = Counter(df_desc.head(window_size)[num_cols].values.flatten()).most_common()
            hot_8 = [int(x[0]) for x in recent_counts[:8]]
            cold_4 = [int(x[0]) for x in recent_counts[-4:]]
            ai_12_pick = sorted(hot_8 + cold_4)
            
            # 介面顯示
            ana_col1, ana_col2 = st.columns([1.2, 2])
            
            with ana_col1:
                st.markdown("#### 🎯 AI 推薦 12 字大底")
                st.markdown(f"""
                <div style='background: linear-gradient(45deg, #FF6B35, #F7931E); padding: 25px; border-radius: 20px; text-align: center;'>
                    <h1 style='color: white; letter-spacing: 5px; font-size: 2.5em;'>
                        {' '.join(map(str, ai_12_pick[:6]))}<br>
                        {' '.join(map(str, ai_12_pick[6:]))}
                    </h1>
                </div>
                """, unsafe_allow_html=True)
                
                fig_g = go.Figure(go.Indicator(mode="gauge+number", value=next_sum, 
                                               title={'text': "預測總和指標 (Sum Index)"},
                                               gauge={'bar':{'color':"#FF6B35"}, 'axis': {'range': [21, 279]}}))
                fig_g.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_g, use_container_width=True)

            with ana_col2:
                st.markdown("#### 📝 AI 深度分析報告")
                with st.container():
                    st.markdown(f"""
                    <div class='ai-analysis-box'>
                        <p><b>1. 總和趨勢分析：</b><br>
                        Prophet 預測下期開彩號碼總和約為 <b>{next_sum:.1f}</b> (範圍: {next_sum_lower:.0f} - {next_sum_upper:.0f})。
                        這代表下期選號應傾向於 <b>{'中大數組合' if next_sum > 150 else '中小數組合'}</b>。</p>
                        
                        <p><b>2. 選號策略說明 (8熱 + 4冷)：</b><br>
                        - <b>熱門字 (動量策略)：</b> 選擇了過去 {window_size} 期最強的 8 個號碼。這類號碼在短期內具有「慣性」，出現頻率較高。<br>
                        - <b>冷門字 (回歸策略)：</b> 選擇了最久未出現的 4 個號碼。基於機率回歸原理，這些號碼「欠債」已久，近期反彈機會增加。</p>
                        
                        <p><b>3. 分佈平衡檢查：</b><br>
                        - <b>單雙比：</b> {' '.join(['單' if x%2!=0 else '雙' for x in ai_12_pick]).count('單')}單 : {' '.join(['單' if x%2!=0 else '雙' for x in ai_12_pick]).count('雙')}雙<br>
                        - <b>大小比：</b> {' '.join(['大' if x>24 else '小' for x in ai_12_pick]).count('大')}大 : {' '.join(['大' if x>24 else '小' for x in ai_12_pick]).count('小')}小<br>
                        <small>*註：AI 建議分佈應盡量接近 1:1 以符合隨機分佈規律。</small></p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.info("⚠️ 聲明：AI 分析僅供參考，並不保證中獎。博彩應適可而止。")

# --- 3. 準確度回測實驗室 ---
if show_backtest:
    st.write("---")
    st.markdown("<h3 class='section-header'>📈 AI 準確度回測實驗室 (模擬 18 字)</h3>", unsafe_allow_html=True)
    test_options = ["最近 100 期", "最近 500 期", "全歷史紀錄"]
    test_depth_sel = st.selectbox("選擇回測範圍", test_options, index=0)
    start_idx = 50 if test_depth_sel == "全歷史紀錄" else (max(50, total_records - 100) if "100" in test_depth_sel else max(50, total_records - 500))
    if st.button("啟動全量回測分析", type="primary"):
        results_log, prizes_count = [], Counter()
        progress_bar = st.progress(0); total_to_test = total_records - start_idx
        for i in range(start_idx, total_records):
            target_row = df_asc.iloc[i]; history_before = df_asc.iloc[i-50:i]
            flat_h = history_before[num_cols].values.flatten()
            freq = Counter(flat_h).most_common()
            ai_18 = set([x[0] for x in freq[:12]] + [x[0] for x in freq[-6:]])
            actual_nums = [int(target_row[n]) for n in num_cols]
            prize_name, rank = calculate_prizes(ai_18, actual_nums, int(target_row['extra']))
            if prize_name:
                prizes_count[prize_name] += 1
                results_log.append({"日期": target_row['date'], "AI 預測 (18字)": sorted(list(ai_18)), "結果": prize_name})
            if (i - start_idx) % 50 == 0: progress_bar.progress((i - start_idx) / total_to_test)
        progress_bar.empty()
        st.success(f"回測完成！共測試 {total_to_test} 期。")
        res_col1, res_col2 = st.columns([1, 2])
        with res_col1:
            st.write("**獲獎匯總:**")
            for p in ["1st Prize", "2nd Prize", "3rd Prize", "4th Prize", "5th Prize", "6th Prize", "7th Prize"]:
                if prizes_count[p] > 0: st.write(f"- {p}: {prizes_count[p]} 次")
        with res_col2:
            if results_log: st.dataframe(pd.DataFrame(results_log), use_container_width=True)

# --- 4. 中獎檢查器 ---
if show_checker:
    st.write("---")
    h_col1, h_col2 = st.columns([5, 1])
    with h_col1: st.markdown("<h3 class='section-header'>🔎 歷史中獎檢查器</h3>", unsafe_allow_html=True)
    with h_col2:
        if st.button("重置選擇", use_container_width=True):
            st.session_state.selected_nums = set(); st.rerun()
    grid = st.columns(7)
    for i in range(1, 50):
        with grid[(i-1) % 7]:
            sel = i in st.session_state.selected_nums
            if st.button(f"{i:02d}", key=f"grid_{i}", use_container_width=True, type="primary" if sel else "secondary"):
                if sel: st.session_state.selected_nums.remove(i)
                elif len(st.session_state.selected_nums) < 6: st.session_state.selected_nums.add(i)
                st.rerun()
    selected_list = sorted(list(st.session_state.selected_nums))
    if len(selected_list) == 6:
        scols = st.columns(3)
        for idx in range(3):
            with scols[idx]:
                if st.button(f"儲存至位置 {idx+1}", key=f"save_{idx}", use_container_width=True):
                    st.session_state.fav_sets[idx] = selected_list
                    sync_favs_to_url(); st.toast("已儲存！"); st.rerun()
        h_res = []
        for _, row in df_desc.iterrows():
            p, r = calculate_prizes(selected_list, [row[n] for n in num_cols], row['extra'])
            if p: h_res.append({"Date": row['date'], "Prize": p, "Rank": r})
        if h_res:
            st.success(f"🎉 歷史上共中獎 **{len(h_res)}** 次")
            st.dataframe(pd.DataFrame(h_res).sort_values("Rank"), use_container_width=True, hide_index=True)

# --- 5. 分析圖表 ---
if show_analysis:
    st.write("---")
    t1, t2 = st.tabs(["📊 出字頻率", "📈 總和趨勢"])
    with t1:
        recent = df_desc.head(window_size)
        all_draws = recent[num_cols].values.flatten()
        freq_df = pd.DataFrame(Counter(all_draws).items(), columns=['Number', 'Count']).sort_values('Count', ascending=False)
        st.plotly_chart(px.bar(freq_df.head(25), x='Number', y='Count', color='Count', color_continuous_scale='Oranges', template="plotly_dark"), use_container_width=True)
    with t2:
        df_desc['draw_sum'] = df_desc[num_cols].sum(axis=1)
        st.plotly_chart(px.area(df_desc.head(window_size), x='date_parsed', y='draw_sum', template="plotly_dark", color_discrete_sequence=['#FF6B35']), use_container_width=True)
