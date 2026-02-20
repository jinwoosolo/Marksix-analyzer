import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from collections import Counter
import datetime

# --- 雲端數據持久化 (Firestore) ---
try:
    from google.cloud import firestore
    db = firestore.Client()
    HAS_CLOUD = True
except:
    HAS_CLOUD = False

APP_ID = "marksix-analyzer"

def get_cloud_ref(code):
    """獲取雲端 Firestore 引用"""
    return db.collection("artifacts").document(APP_ID).collection("public").document("data").collection("user_favs").document(code)

def cloud_save():
    """保存組合至雲端"""
    if HAS_CLOUD and st.session_state.sync_code:
        ref = get_cloud_ref(st.session_state.sync_code)
        data = {"favs": [list(f) if f else None for f in st.session_state.fav_sets]}
        ref.set(data)
        st.toast(f"✅ 雲端同步成功 (同步碼: {st.session_state.sync_code})")

def cloud_load():
    """從雲端載入組合"""
    if HAS_CLOUD and st.session_state.sync_code:
        ref = get_cloud_ref(st.session_state.sync_code)
        doc = ref.get()
        if doc.exists:
            data = doc.to_dict()
            st.session_state.fav_sets = [set(f) if f else None for f in data.get("favs", [None, None, None])]
            st.toast("📂 已成功從雲端回復組合")
        else:
            st.toast("ℹ️ 此同步碼目前沒有雲端紀錄")

# --- 核心數據邏輯 ---
@st.cache_data(ttl=3600)
def load_data():
    """加載原始數據"""
    df = pd.read_csv('marksix.csv')
    df['date_parsed'] = pd.to_datetime(df['date'], format='mixed')
    return df.sort_values('date_parsed', ascending=True).reset_index(drop=True)

@st.cache_data(show_spinner=False)
def get_all_historical_wins_optimized(user_set_tuple, total_records_count):
    """
    優化版：計算中獎紀錄。
    不再接收巨大的 JSON，改為接收總紀錄數作為緩存 Key 的一部分。
    """
    # 直接調用已經緩存過的 load_data
    df = load_data() 
    user_set = set(user_set_tuple)
    results = []
    
    # 遍歷計算
    for _, row in df.iterrows():
        draw_set = {row['n1'], row['n2'], row['n3'], row['n4'], row['n5'], row['n6']}
        matched = user_set.intersection(draw_set)
        m = len(matched)
        e = int(row['extra']) in user_set
        
        prize, rank = None, 99
        if m == 6: prize, rank = "1st Prize", 1
        elif m == 5 and e: prize, rank = "2nd Prize", 2
        elif m == 5: prize, rank = "3rd Prize", 3
        elif m == 4 and e: prize, rank = "4th Prize", 4
        elif m == 4: prize, rank = "5th Prize", 5
        elif m == 3 and e: prize, rank = "6th Prize", 6
        elif m == 3: prize, rank = "7th Prize", 7
        
        if prize:
            results.append({"Date": row['date'], "Prize": prize, "Rank": rank})
            
    return results

# --- APP CONFIGURATION ---
st.set_page_config(page_title="六合彩 AI 專業分析器 Pro", page_icon="🎰", layout="wide")

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
    .date-list { font-size: 0.75em; color: #aaa; margin-top: 5px; line-height: 1.4; }
    .section-header { margin-top: 15px; margin-bottom: 15px; font-weight: bold; border-left: 5px solid #FF6B35; padding-left: 10px; }
    .ai-analysis-box { background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 107, 53, 0.2); border-radius: 12px; padding: 15px; color: #e0e0e0; line-height: 1.6; }
    </style>
    """, unsafe_allow_html=True)

try:
    df_asc = load_data()
    df_desc = df_asc.sort_values('date_parsed', ascending=False).reset_index(drop=True)
    total_records = len(df_asc)
    num_cols = ['n1', 'n2', 'n3', 'n4', 'n5', 'n6']
except Exception as e:
    st.error(f"⚠️ 數據載入錯誤: {e}"); st.stop()

# --- INITIALIZE STATE ---
if 'fav_sets' not in st.session_state: st.session_state.fav_sets = [None, None, None]
if 'selected_nums' not in st.session_state: st.session_state.selected_nums = set()
if 'sync_code' not in st.session_state: st.session_state.sync_code = ""

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ 控制面板")
    st.subheader("☁️ 跨裝置同步")
    s_code = st.text_input("輸入私密同步碼", placeholder="例如: my-secret-sets")
    if s_code and s_code != st.session_state.sync_code:
        st.session_state.sync_code = s_code
        cloud_load()

    st.divider()
    st.subheader("顯示設定")
    v_fav = st.checkbox("顯示組合追蹤", value=True)
    v_ai = st.checkbox("顯示 AI 預測專區", value=True)
    v_check = st.checkbox("顯示中獎檢查器", value=True)
    v_chart = st.checkbox("顯示分析圖表", value=True)
    v_test = st.checkbox("顯示回測實驗室", value=False)
    st.divider()
    window = st.slider("統計窗口 (期數)", 10, 500, 100)

# --- DASHBOARD HEADER ---
latest = df_desc.iloc[0]
st.title("🎰 六合彩 AI 專業分析器 Pro")
c_m1, c_m2, c_m3 = st.columns(3)
with c_m1: st.markdown(f"<div class='metric-card'><span class='stat-label'>最近日期</span><br><span class='stat-val'>{latest['date']}</span></div>", unsafe_allow_html=True)
with c_m2: st.markdown(f"<div class='metric-card'><span class='stat-label'>中獎號碼</span><br><span class='stat-val'>{'  '.join([str(int(latest[n])) for n in num_cols])}</span></div>", unsafe_allow_html=True)
with c_m3: st.markdown(f"<div class='metric-card'><span class='stat-label'>特別號碼</span><br><span class='stat-val' style='color:#7FD1B9'>{int(latest['extra'])}</span></div>", unsafe_allow_html=True)

# --- 1. FAVORITE SETS TRACKER ---
if v_fav:
    st.markdown("<h3 class='section-header'>⭐ 我的最愛組合追蹤 (雲端同步)</h3>", unsafe_allow_html=True)
    f_cols = st.columns(3)
    for i, fav in enumerate(st.session_state.fav_sets):
        with f_cols[i]:
            if fav:
                st.markdown("<div class='favorite-card'>", unsafe_allow_html=True)
                st.markdown(f"**位置 {i+1}**")
                sorted_fav = sorted(list(fav))
                st.code(" ".join(map(str, sorted_fav)))
                
                # 使用優化後的緩存計算
                f_res = get_all_historical_wins_optimized(tuple(sorted_fav), total_records)
                
                high = [r for r in f_res if r['Rank'] <= 4]
                latest_nums = {int(latest[n]) for n in num_cols}
                l_match = len(set(fav).intersection(latest_nums))
                
                st.markdown(f"總中獎: **{len(f_res)}** | 大獎(4獎+): <span class='win-highlight'>{len(high)}</span>", unsafe_allow_html=True)
                if high: st.markdown(f"<div class='date-list'><b>大獎日期:</b><br>{', '.join([r['Date'] for r in high])}</div>", unsafe_allow_html=True)
                if l_match >= 3: st.warning(f"🔔 最新一期中獎！")
                
                if st.button(f"清除 Slot {i+1}", key=f"clr_{i}", width='stretch'):
                    st.session_state.fav_sets[i] = None
                    cloud_save(); st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.info(f"Slot {i+1} 空位")

# --- 2. AI 智能預測 ---
if v_ai:
    st.write("---")
    st.markdown("<h3 class='section-header'>🔮 AI 智能預測與深度分析 (下期推薦 12 字)</h3>", unsafe_allow_html=True)
    if st.button("🚀 生成下期 AI 智能深度分析報告", type="primary", width='stretch'):
        with st.spinner("模型分析中..."):
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
                st.markdown(f"""<div style='background: linear-gradient(45deg, #FF6B35, #F7931E); padding: 25px; border-radius: 15px; text-align: center; color: white;'><h1 style='font-size: 2.2em; letter-spacing: 5px;'>{' '.join(map(str, ai_12[:6]))}<br>{' '.join(map(str, ai_12[6:]))}</h1></div>""", unsafe_allow_html=True)
                fig_g = go.Figure(go.Indicator(mode="gauge+number", value=n_sum, title={'text': "下期總和預測"}, gauge={'bar':{'color':"#FF6B35"}, 'axis':{'range':[21,279]}}))
                fig_g.update_layout(height=230, margin=dict(l=10, r=10, t=40, b=0), paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_g, width='stretch')
            with a_c2:
                st.markdown("#### 📝 預測邏輯分析")
                st.markdown(f"""<div class="ai-analysis-box"><strong>1. 總和趨勢分析：</strong> Prophet 模型預測下期數字總和約為 <strong>{n_sum:.1f}</strong>。代表選號應偏向於 <strong>{"大數組合" if n_sum > 150 else "小數組合"}</strong>。<br><br><strong>2. 選號策略 (8熱 + 4冷)：</strong> 結合最強動量 8 熱門號碼與 4 個遺漏回歸冷門號碼。<br><br><strong>3. 平衡檢查：</strong> {len([x for x in ai_12 if x%2!=0])}單 : {len([x for x in ai_12 if x%2==0])}雙 | {len([x for x in ai_12 if x>24])}大 : {len([x for x in ai_12 if x<=24])}小</div>""", unsafe_allow_html=True)

# --- 3. 中獎檢查器 (OPTIMIZED) ---
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
        st.write("💾 **儲存至雲端位置:**")
        sv_cols = st.columns(3)
        for i in range(3):
            with sv_cols[i]:
                if st.button(f"存入位置 {i+1}", key=f"sv_{i}", width='stretch'):
                    st.session_state.fav_sets[i] = set(sl)
                    cloud_save(); st.rerun()
        # 關鍵優化：不再在這裡進行耗時計算，使用緩存過的函數
        h_res = get_all_historical_wins_optimized(tuple(sl), total_records)
        if h_res:
            st.success(f"🎉 歷史共中獎 {len(h_res)} 次")
            st.dataframe(pd.DataFrame(h_res).sort_values("Rank"), width='stretch', hide_index=True)
        else:
            st.info("此組合在歷史中未曾獲獎。")

# --- 4. 準確度回測 ---
if v_test:
    st.write("---")
    st.markdown("<h3 class='section-header'>📈 AI 準確度回測實驗室 (12字策略模擬)</h3>", unsafe_allow_html=True)
    scope = st.selectbox("選擇回測範圍", ["最近 100 期", "最近 500 期", "全歷史紀錄"])
    s_idx = 50 if "全" in scope else (total_records-100 if "100" in scope else total_records-500)
    if st.button("執行 12 字策略回測", width='stretch'):
        log, pc = [], Counter()
        prog = st.progress(0)
        for i in range(max(50, s_idx), total_records):
            tar = df_asc.iloc[i]; hist = df_asc.iloc[i-50:i]
            fq = Counter(hist[num_cols].values.flatten()).most_common()
            ai_12_back = set([x[0] for x in fq[:8]] + [x[0] for x in fq[-4:]])
            
            # 直接計算單次中獎
            matched = ai_12_back.intersection({int(tar[n]) for n in num_cols})
            m_count = len(matched)
            has_e = int(tar['extra']) in ai_12_back
            p_name = None
            if m_count == 6: p_name = "1st Prize"
            elif m_count == 5 and has_e: p_name = "2nd Prize"
            elif m_count == 5: p_name = "3rd Prize"
            elif m_count == 4 and has_e: p_name = "4th Prize"
            elif m_count == 4: p_name = "5th Prize"
            elif m_count == 3 and has_e: p_name = "6th Prize"
            elif m_count == 3: p_name = "7th Prize"
            
            if p_name:
                pc[p_name] += 1
                log.append({"日期": tar['date'], "AI 預測(12字)": sorted(list(ai_12_back)), "當期開彩": f"{[int(tar[n]) for n in num_cols]} + ({int(tar['extra'])})", "結果": p_name})
            prog.progress((i-s_idx)/(total_records-s_idx))
        st.write("**回測獲獎統計匯總:**")
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
