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
    .section-header { margin-top: 10px; margin-bottom: 15px; font-weight: bold; border-left: 5px solid #FF6B35; padding-left: 10px; }
    .ai-analysis-box {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 107, 53, 0.2);
        border-radius: 12px;
        padding: 15px;
        color: #e0e0e0;
        line-height: 1.6;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 持久化邏輯 ---
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

# --- 側邊欄 ---
with st.sidebar:
    st.title("⚙️ 控制面板")
    show_favs = st.checkbox("顯示組合追蹤", value=True)
    show_ai_predict = st.checkbox("顯示 AI 預測專區", value=True)
    show_checker = st.checkbox("顯示中獎檢查器", value=True)
    show_analysis = st.checkbox("顯示分析圖表", value=True)
    show_backtest = st.checkbox("顯示回測實驗室", value=False)
    st.divider()
    window_size = st.slider("統計期數 (窗口)", 10, 500, 100)
    st.info(f"📊 總紀錄: {total_records} 期")

# --- 頁首 ---
st.title("🎰 六合彩 AI 專業分析器 Pro")
latest = df_desc.iloc[0]
l_nums = "  ".join([str(int(latest[n])) for n in num_cols])

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f"<div class='metric-card'><span class='stat-label'>最近開彩</span><br><span class='stat-val'>{latest['date']}</span></div>", unsafe_allow_html=True)
with c2:
    st.markdown(f"<div class='metric-card'><span class='stat-label'>中獎號碼</span><br><span class='stat-val'>{l_nums}</span></div>", unsafe_allow_html=True)
with c3:
    st.markdown(f"<div class='metric-card'><span class='stat-label'>特別號碼</span><br><span class='stat-val' style='color:#7FD1B9'>{int(latest['extra'])}</span></div>", unsafe_allow_html=True)

# --- 1. 收藏組合 ---
if show_favs:
    st.markdown("<h3 class='section-header'>⭐ 最愛組合即時追蹤</h3>", unsafe_allow_html=True)
    fav_cols = st.columns(3)
    for idx, fav in enumerate(st.session_state.fav_sets):
        with fav_cols[idx]:
            if fav:
                st.markdown("<div class='favorite-card'>", unsafe_allow_html=True)
                st.markdown(f"**Slot {idx+1}**")
                st.code(" ".join(map(str, fav)))
                
                fav_res = []
                for _, row in df_desc.iterrows():
                    p, r = calculate_prizes(fav, [row[n] for n in num_cols], row['extra'])
                    if p: fav_res.append({"Date": row['date'], "Prize": p, "Rank": r})
                
                high = [r for r in fav_res if r['Rank'] <= 4]
                latest_match = len(set(fav).intersection({latest[n] for n in num_cols}))
                
                st.markdown(f"總中獎: **{len(fav_res)}** | 大獎: <span class='win-highlight'>{len(high)}</span>", unsafe_allow_html=True)
                if high: st.markdown(f"<div class='date-list'>日期: {', '.join([r['Date'] for r in high])}</div>", unsafe_allow_html=True)
                if latest_match >= 3: st.warning(f"🔔 最新一期中 {latest_match} 字!")
                
                if st.button(f"清除 Slot {idx+1}", key=f"clr_{idx}", use_container_width=True):
                    st.session_state.fav_sets[idx] = None
                    sync_favs_to_url(); st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.info(f"Slot {idx+1} 空位")

# --- 2. AI 智能預測專區 ---
if show_ai_predict:
    st.write("---")
    st.markdown("<h3 class='section-header'>🔮 AI 智能預測與深度分析 (下期推薦 12 字)</h3>", unsafe_allow_html=True)
    
    if st.button("🚀 生成下期 AI 預測分析報告", type="primary", use_container_width=True):
        with st.spinner("融合 Prophet 模型分析中..."):
            # 1. Prophet 預測
            df_desc['draw_sum'] = df_desc[num_cols].sum(axis=1)
            p_df = df_desc.rename(columns={'date_parsed': 'ds', 'draw_sum': 'y'})[['ds', 'y']].dropna()
            m = Prophet(daily_seasonality=False, weekly_seasonality=True)
            m.fit(p_df)
            forecast = m.predict(m.make_future_dataframe(periods=1, freq='3D'))
            next_sum = forecast['yhat'].iloc[-1]
            
            # 2. 8熱 + 4冷 邏輯
            recent_counts = Counter(df_desc.head(window_size)[num_cols].values.flatten()).most_common()
            hot_8 = [int(x[0]) for x in recent_counts[:8]]
            cold_4 = [int(x[0]) for x in recent_counts[-4:]]
            ai_12 = sorted(hot_8 + cold_4)
            
            # 顯示
            a_col1, a_col2 = st.columns([1, 1.5])
            with a_col1:
                st.markdown("#### 🎯 AI 推薦 12 字大底")
                st.markdown(f"""
                <div style='background: linear-gradient(45deg, #FF6B35, #F7931E); padding: 30px; border-radius: 15px; text-align: center; color: white;'>
                    <h1 style='font-size: 2.2em; letter-spacing: 5px;'>{' '.join(map(str, ai_12[:6]))}<br>{' '.join(map(str, ai_12[6:]))}</h1>
                </div>
                """, unsafe_allow_html=True)
                
                fig_g = go.Figure(go.Indicator(mode="gauge+number", value=next_sum, title={'text': "下期總和預測"},
                                               gauge={'bar':{'color':"#FF6B35"}, 'axis':{'range':[21,279]}}))
                fig_g.update_layout(height=250, margin=dict(l=10, r=10, t=50, b=0), paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_g, use_container_width=True)

            with a_col2:
                st.markdown("#### 📝 AI 預測邏輯分析")
                odd_count = len([x for x in ai_12 if x % 2 != 0])
                big_count = len([x for x in ai_12 if x > 24])
                
                # 直接使用 Markdown 渲染，避免 HTML 渲染失敗
                st.markdown(f"""
                <div class="ai-analysis-box">
                    <strong>1. 總和趨勢分析：</strong><br>
                    Prophet 模型預測下期數字總和約為 <strong>{next_sum:.1f}</strong>。根據歷史數據，這意味著選號應偏向於 <strong>{"大數組合" if next_sum > 150 else "小數組合"}</strong>。
                    <br><br>
                    <strong>2. 選號策略說明 (8熱 + 4冷)：</strong><br>
                    - <strong>熱門字 (動量策略)：</strong> 選擇了過去 {window_size} 期最強的 8 個號碼。這類號碼在短期內具有規律性，出現頻率較高。<br>
                    - <strong>冷門字 (回歸策略)：</strong> 選擇了最久未出現的 4 個號碼。基於機率回歸原理，這些號碼近期反彈機會增加。
                    <br><br>
                    <strong>3. 數據平衡檢查：</strong><br>
                    - <strong>單雙比：</strong> {odd_count}單 : {12-odd_count}雙<br>
                    - <strong>大小比：</strong> {big_count}大 : {12-big_count}小<br>
                    <small>*提示：理想組合應接近 1:1 分佈以符合隨機規律。</small>
                </div>
                """, unsafe_allow_html=True)

# --- 3. 中獎檢查器 ---
if show_checker:
    st.write("---")
    h1, h2 = st.columns([5, 1])
    with h1: st.markdown("<h3 class='section-header'>🔎 歷史中獎檢查器</h3>", unsafe_allow_html=True)
    with h2:
        if st.button("重置", use_container_width=True): st.session_state.selected_nums = set(); st.rerun()

    grid = st.columns(7)
    for i in range(1, 50):
        with grid[(i-1)%7]:
            sel = i in st.session_state.selected_nums
            if st.button(f"{i:02d}", key=f"g_{i}", use_container_width=True, type="primary" if sel else "secondary"):
                if sel: st.session_state.selected_nums.remove(i)
                elif len(st.session_state.selected_nums) < 6: st.session_state.selected_nums.add(i)
                st.rerun()
    
    sl = sorted(list(st.session_state.selected_nums))
    if len(sl) == 6:
        sc = st.columns(3)
        for i in range(3):
            with sc[i]:
                if st.button(f"存入位置 {i+1}", key=f"sv_{i}", use_container_width=True):
                    st.session_state.fav_sets[i] = sl
                    sync_favs_to_url(); st.toast("儲存成功！"); st.rerun()
        
        hr = []
        for _, row in df_desc.iterrows():
            p, r = calculate_prizes(sl, [row[n] for n in num_cols], row['extra'])
            if p: hr.append({"Date": row['date'], "Prize": p, "Rank": r})
        if hr:
            st.success(f"🎉 歷史中共中獎 {len(hr)} 次")
            st.dataframe(pd.DataFrame(hr).sort_values("Rank"), use_container_width=True, hide_index=True)

# --- 4. 回測實驗室 ---
if show_backtest:
    st.write("---")
    st.markdown("<h3 class='section-header'>📈 AI 準確度回測 (模擬 18 字大底)</h3>", unsafe_allow_html=True)
    depth = st.selectbox("回測範圍", ["最近 100 期", "最近 500 期", "全歷史紀錄"])
    s_idx = 50 if "全" in depth else (total_records-100 if "100" in depth else total_records-500)
    if st.button("執行回測"):
        log, pc = [], Counter()
        prog = st.progress(0)
        for i in range(max(50, s_idx), total_records):
            tar = df_asc.iloc[i]; hist = df_asc.iloc[i-50:i]
            fq = Counter(hist[num_cols].values.flatten()).most_common()
            ai_18 = set([x[0] for x in fq[:12]] + [x[0] for x in fq[-6:]])
            p, r = calculate_prizes(ai_18, [tar[n] for n in num_cols], int(tar['extra']))
            if p:
                pc[p] += 1
                log.append({"日期": tar['date'], "開彩": f"{[int(tar[n]) for n in num_cols]} + ({int(tar['extra'])})", "結果": p})
            prog.progress((i-s_idx)/(total_records-s_idx))
        st.write(pc); st.dataframe(pd.DataFrame(log))

# --- 5. 圖表 ---
if show_analysis:
    st.write("---")
    t1, t2 = st.tabs(["📊 出字頻率", "📈 總和趨勢"])
    with t1:
        fq = pd.DataFrame(Counter(df_desc.head(window_size)[num_cols].values.flatten()).items(), columns=['No','Count']).sort_values('Count', ascending=False)
        st.plotly_chart(px.bar(fq.head(25), x='No', y='Count', color='Count', color_continuous_scale='Oranges', template="plotly_dark"), use_container_width=True)
    with t2:
        df_desc['draw_sum'] = df_desc[num_cols].sum(axis=1)
        st.plotly_chart(px.area(df_desc.head(window_size), x='date_parsed', y='draw_sum', template="plotly_dark", color_discrete_sequence=['#FF6B35']), use_container_width=True)
