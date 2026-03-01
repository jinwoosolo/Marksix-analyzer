import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import os
import time
from urllib.parse import urljoin
from datetime import datetime

# --- 全域配置 ---
BASE_URL = "https://www.lottery.hk"
RESULTS_LIST_URL = "https://www.lottery.hk/en/mark-six/results"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
}

def parse_draw_page(html, url):
    """
    🔥 解析詳情頁邏輯：確保抓到正選與特別號碼
    """
    if not html:
        return {}

    soup = BeautifulSoup(html, "lxml")
    data = {"url": url}

    # 1. 提取日期 (從 URL 提取)
    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", url)
    if date_match:
        data["date"] = date_match.group(1).replace("-", "/")
    
    # 2. 提取號碼
    balls = []
    # 鎖定結果容器
    target_container = soup.select_one(".result-box") or soup.select_one(".draw-results") or soup
    
    ball_elements = target_container.find_all(['li', 'span', 'div'], class_=re.compile(r'ball|no-|num', re.I))
    for el in ball_elements:
        txt = el.get_text(strip=True)
        if txt.isdigit():
            val = int(txt)
            if 1 <= val <= 49 and val not in balls:
                balls.append(val)

    # 3. 數據分配
    if len(balls) >= 7:
        main = balls[:6]
        extra = balls[6]
        data.update({
            "n1": main[0], "n2": main[1], "n3": main[2],
            "n4": main[3], "n5": main[4], "n6": main[5],
            "extra": extra
        })
    
    # 4. 獎金
    full_text = soup.get_text()
    prize_match = re.search(r'1st.*?HK\$?([\d,]+)', full_text, re.I | re.DOTALL)
    data["div1_prize"] = prize_match.group(1).replace(',', '') if prize_match else "0"
    data["div1_winners"] = "0"

    return data

def fetch_latest_marksix():
    """
    主執行程序：具備「追溯功能」，會自動補齊 CSV 中缺失的近期數據
    """
    print(f"\n{'='*50}")
    print(f"🚀 六合彩追溯抓取器啟動: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*50}")

    try:
        # 載入現有 CSV 以進行比對
        csv_path = 'marksix.csv'
        if os.path.exists(csv_path):
            df_existing = pd.read_csv(csv_path)
            df_existing['date'] = df_existing['date'].astype(str)
            existing_dates = set(df_existing['date'].values)
        else:
            df_existing = pd.DataFrame()
            existing_dates = set()

        # 1. 存取列表頁
        resp = requests.get(RESULTS_LIST_URL, headers=headers, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        
        # 2. 找出列表頁上所有的近期連結 (而不僅僅是第一個)
        draw_links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if re.search(r"/mark-six/results/20\d{2}-\d{2}-\d{2}", href):
                full_url = urljoin(BASE_URL, href)
                date_in_url = re.search(r"(\d{4}-\d{2}-\d{2})", href).group(1).replace("-", "/")
                # 只記錄尚未存在於 CSV 中的日期
                if date_in_url not in existing_dates:
                    draw_links.append((date_in_url, full_url))

        if not draw_links:
            print("ℹ️ 所有近期數據已存在於 CSV 中，無需更新。")
            return True
            
        print(f"🔎 發現 {len(draw_links)} 筆缺失或需要更新的數據。")

        # 3. 逐一抓取缺失的數據 (按日期順序，從舊到新)
        new_records = []
        for missing_date, link in reversed(draw_links): # 使用 reversed 確保從舊日期開始補
            print(f"⏳ 正在補齊日期: {missing_date} ...")
            time.sleep(2) 
            detail_resp = requests.get(link, headers=headers, timeout=20)
            if detail_resp.status_code == 200:
                rec = parse_draw_page(detail_resp.text, link)
                if rec.get("n1"):
                    new_records.append(rec)
                    print(f"   ✅ 成功抓取: {rec['n1']}, {rec['n2']}, {rec['n3']}, {rec['n4']}, {rec['n5']}, {rec['n6']} + {rec['extra']}")

        # 4. 合併並儲存
        if new_records:
            df_new = pd.DataFrame(new_records)
            df_final = pd.concat([df_existing, df_new], ignore_index=True)
            
            # 清洗重複項並排序
            df_final['date_dt'] = pd.to_datetime(df_final['date'], format='%Y/%m/%d')
            df_final = df_final.sort_values('date_dt', ascending=True).drop_duplicates(subset=['date']).drop(columns=['date_dt'])
            
            df_final.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"\n🎉 更新完成！CSV 現在包含 {len(df_final)} 筆紀錄。")
            return True
        
    except Exception as e:
        print(f"❌ 發生錯誤: {e}")
    
    return False

if __name__ == "__main__":
    fetch_latest_marksix()
