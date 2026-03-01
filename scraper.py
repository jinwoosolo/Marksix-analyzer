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
    🔥 強化版解析邏輯：確保在詳情頁抓到正確的 6+1 號碼
    """
    if not html:
        return {}

    soup = BeautifulSoup(html, "lxml")
    data = {"url": url}

    # 1. 提取日期 (從 URL 提取是最準確的)
    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", url)
    if date_match:
        data["date"] = date_match.group(1).replace("-", "/")
    
    # 2. 提取號碼
    balls = []
    
    # 策略 A: 鎖定特定結果容器 (針對不同頁面模板)
    container = soup.select_one(".result-box") or \
                soup.select_one(".draw-results") or \
                soup.select_one(".results-table") or \
                soup.select_one("main")
    
    if not container:
        container = soup

    # 尋找所有具有球類特徵的元素
    # lottery.hk 常用 class: ball, no-xx, r-b, g-b, b-b
    ball_elements = container.find_all(['li', 'span', 'div'], class_=re.compile(r'ball|no-|num|result-num', re.I))
    
    for el in ball_elements:
        txt = el.get_text(strip=True)
        if txt.isdigit():
            val = int(txt)
            if 1 <= val <= 49 and val not in balls:
                balls.append(val)

    # 策略 B: 如果標籤抓不到，嘗試 Regex 掃描 (排除日期中的數字)
    if len(balls) < 7:
        # 獲取純文本並排除 URL 中的日期干擾
        text = container.get_text(separator=' ', strip=True)
        date_nums = set(re.findall(r'\d+', data.get("date", "")))
        potential_nums = re.findall(r'\b\d{1,2}\b', text)
        for n in potential_nums:
            if n.isdigit():
                val = int(n)
                if 1 <= val <= 49 and n not in date_nums and val not in balls:
                    balls.append(val)

    # 3. 數據分配
    if len(balls) >= 7:
        # 前 6 個是正獎，第 7 個是特別號
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
    主執行程序：自動補齊 CSV 中缺失的數據
    """
    print(f"\n{'='*60}")
    print(f"🚀 六合彩追溯抓取器啟動: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    try:
        # 載入現有 CSV
        csv_path = 'marksix.csv'
        if os.path.exists(csv_path):
            df_existing = pd.read_csv(csv_path)
            df_existing['date'] = df_existing['date'].astype(str)
            existing_dates = set(df_existing['date'].values)
        else:
            df_existing = pd.DataFrame(columns=['date', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'extra', 'div1_prize', 'div1_winners', 'url'])
            existing_dates = set()

        # 1. 獲取列表頁
        resp = requests.get(RESULTS_LIST_URL, headers=headers, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        
        # 2. 搜尋所有開獎連結
        draw_links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "/mark-six/results/" in href:
                # 提取日期 YYYY-MM-DD
                match = re.search(r"(\d{4}-\d{2}-\d{2})", href)
                if match:
                    date_val = match.group(1).replace("-", "/")
                    if date_val not in existing_dates:
                        full_url = urljoin(BASE_URL, href)
                        draw_links.append((date_val, full_url))

        if not draw_links:
            print("ℹ️ CSV 數據已是最新，無需抓取。")
            return True
            
        print(f"🔎 發現 {len(draw_links)} 筆缺失數據。")

        # 3. 逐一補齊
        new_records = []
        for missing_date, link in reversed(draw_links): # 從舊到新
            print(f"⏳ 正在補齊: {missing_date} ...")
            time.sleep(2) # 禮貌延遲
            try:
                detail_resp = requests.get(link, headers=headers, timeout=20)
                if detail_resp.status_code == 200:
                    rec = parse_draw_page(detail_resp.text, link)
                    if rec.get("n1"):
                        new_records.append(rec)
                        print(f"   ✅ 成功: {rec['n1']}, {rec['n2']}, {rec['n3']}, {rec['n4']}, {rec['n5']}, {rec['n6']} + {rec['extra']}")
                    else:
                        print(f"   ⚠️ 無法解析號碼數據。")
            except Exception as e:
                print(f"   ❌ 抓取失敗: {e}")

        # 4. 合併並保存
        if new_records:
            df_new = pd.DataFrame(new_records)
            df_final = pd.concat([df_existing, df_new], ignore_index=True)
            
            # 數據整理
            df_final['date_dt'] = pd.to_datetime(df_final['date'], format='%Y/%m/%d')
            df_final = df_final.sort_values('date_dt', ascending=True)
            df_final = df_final.drop_duplicates(subset=['date'], keep='last').drop(columns=['date_dt'])
            
            df_final.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"\n🎉 更新完成！目前 CSV 總量: {len(df_final)} 筆。")
            return True
        else:
            print("❌ 未能成功獲取任何新紀錄。")
        
    except Exception as e:
        print(f"❌ 嚴重錯誤: {e}")
    
    return False

if __name__ == "__main__":
    fetch_latest_marksix()
