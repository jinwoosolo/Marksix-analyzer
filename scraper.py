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
    🔥 終極解析邏輯：利用 CSS 顏色類名精確定位號碼球 (r-ball, g-ball, b-ball)
    """
    if not html:
        return {}

    soup = BeautifulSoup(html, "lxml")
    data = {"url": url}

    # 1. 提取日期 (從 URL 提取)
    date_match = re.search(r"(\d{4})-(\d{2})-(\d{2})", url)
    if date_match:
        year, month, day = date_match.groups()
        data["date"] = f"{year}/{month}/{day}"
    else:
        return {}

    print(f"   [Debug] 處理日期: {data.get('date')}")

    # 2. 定位結果核心區塊
    # 我們鎖定包含 'ball' 類名且為數字的元素
    # lottery.hk 的號碼球 class 通常包含 'r-ball' (紅), 'g-ball' (綠), 'b-ball' (藍)
    balls = []
    
    # 策略 A: 尋找所有帶有顏色球標籤的元素
    # 這是最準確的方法，因為文字標題不會帶有顏色球樣式
    ball_elements = soup.find_all(['span', 'li', 'div'], class_=re.compile(r'r-ball|g-ball|b-ball|ball', re.I))
    
    for el in ball_elements:
        txt = el.get_text(strip=True)
        if txt.isdigit():
            val = int(txt)
            if 1 <= val <= 49 and val not in balls:
                balls.append(val)
    
    # 3. 策略 B: 如果標籤法沒抓到，嘗試鎖定特定的容器
    if len(balls) < 7:
        print("   [Debug] 標籤法未達標，嘗試容器定位...")
        container = soup.select_one(".result-numbers") or \
                    soup.select_one(".balls") or \
                    soup.select_one(".draw-results")
        
        if container:
            nums = container.find_all(['span', 'li', 'div'])
            for n in nums:
                t = n.get_text(strip=True)
                if t.isdigit():
                    v = int(t)
                    if 1 <= v <= 49 and v not in balls:
                        balls.append(v)

    # 4. 數據檢查與分開正獎/特別獎
    # 詳情頁中，如果有 7 個球，前 6 個是正獎，最後一個是特別號
    if len(balls) >= 7:
        # 特別處理：如果抓到太多號碼，排除掉可能是日期或年份的數字
        # 通常詳情頁中心區域的 7 個連續數字才是正確的
        if len(balls) > 7:
            # 只取前 7 個
            balls = balls[:7]
            
        res = balls
        data.update({
            "n1": res[0], "n2": res[1], "n3": res[2],
            "n4": res[3], "n5": res[4], "n6": res[5],
            "extra": res[6]
        })
        print(f"   🎯 解析成功: {res[:6]} + {res[6]}")
    else:
        print(f"   ❌ 解析失敗，僅抓到: {balls}")

    # 5. 獎金提取
    full_text = soup.get_text()
    prize_match = re.search(r'1st.*?HK\$?([\d,]+)', full_text, re.I | re.DOTALL)
    data["div1_prize"] = prize_match.group(1).replace(',', '') if prize_match else "0"
    data["div1_winners"] = "0"

    return data

def fetch_latest_marksix():
    """
    主執行程序：自動補齊 CSV 缺失數據
    """
    print(f"\n{'='*60}")
    print(f"🚀 六合彩精確抓取器啟動: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    try:
        csv_path = 'marksix.csv'
        if os.path.exists(csv_path):
            df_existing = pd.read_csv(csv_path)
            df_existing['date'] = df_existing['date'].astype(str)
            existing_dates = set(df_existing['date'].values)
        else:
            df_existing = pd.DataFrame(columns=['date', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'extra', 'div1_prize', 'div1_winners', 'url'])
            existing_dates = set()

        resp = requests.get(RESULTS_LIST_URL, headers=headers, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        
        draw_links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "/mark-six/results/20" in href:
                match = re.search(r"(\d{4}-\d{2}-\d{2})", href)
                if match:
                    date_val = match.group(1).replace("-", "/")
                    if date_val not in existing_dates:
                        full_url = urljoin(BASE_URL, href)
                        if (date_val, full_url) not in draw_links:
                            draw_links.append((date_val, full_url))
        
        draw_links = draw_links[:5]

        if not draw_links:
            print("ℹ️ CSV 數據已是最新。")
            return True
            
        print(f"🔎 發現 {len(draw_links)} 筆需要補齊的數據...")

        new_records = []
        for missing_date, link in reversed(draw_links):
            print(f"⏳ 正在處理詳情頁: {missing_date} ...")
            time.sleep(2) 
            try:
                detail_resp = requests.get(link, headers=headers, timeout=20)
                if detail_resp.status_code == 200:
                    rec = parse_draw_page(detail_resp.text, link)
                    if rec.get("n1"):
                        new_records.append(rec)
                    else:
                        print(f"   ❌ {missing_date} 解析失敗。")
            except Exception as e:
                print(f"   ❌ 抓取錯誤: {e}")

        if new_records:
            df_new = pd.DataFrame(new_records)
            df_final = pd.concat([df_existing, df_new], ignore_index=True)
            df_final['date_dt'] = pd.to_datetime(df_final['date'], format='%Y/%m/%d')
            df_final = df_final.sort_values('date_dt', ascending=True)
            df_final = df_final.drop_duplicates(subset=['date'], keep='last').drop(columns=['date_dt'])
            df_final.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"\n🎉 數據同步完成！")
            return True
        
    except Exception as e:
        print(f"❌ 嚴重錯誤: {e}")
    
    return False

if __name__ == "__main__":
    fetch_latest_marksix()
