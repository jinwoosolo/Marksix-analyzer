import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import os
import time
from urllib.parse import urljoin
from datetime import datetime

# --- 配置區 ---
BASE_URL = "https://www.lottery.hk"
RESULTS_LIST_URL = "https://www.lottery.hk/en/mark-six/results"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
}

def fetch_latest_marksix():
    """
    精確抓取邏輯：
    1. 進入列表頁找到最新一期的獨立網址。
    2. 進入詳情頁，只鎖定存放號碼球的 HTML 容器。
    3. 排除標題、期數、獎金等雜訊數字。
    """
    try:
        print(f"🚀 正在存取列表頁: {RESULTS_LIST_URL}")
        resp = requests.get(RESULTS_LIST_URL, headers=headers, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        
        # 1. 找到最新一期的詳細連結
        latest_link = None
        for a in soup.find_all("a", href=True):
            href = a["href"]
            # 尋找符合 /en/mark-six/results/YYYY-MM-DD 格式的連結
            if re.search(r"/mark-six/results/20\d{2}-\d{2}-\d{2}", href):
                latest_link = urljoin(BASE_URL, href)
                break
        
        if not latest_link:
            print("❌ 找不到最新一期的詳細連結"); return False
            
        print(f"🔎 進入詳情頁抓取正確數據: {latest_link}")
        
        # 2. 存取詳情頁
        time.sleep(2) # 延遲以確保伺服器響應
        detail_resp = requests.get(latest_link, headers=headers, timeout=20)
        detail_resp.raise_for_status()
        detail_soup = BeautifulSoup(detail_resp.text, "lxml")
        
        # 3. 從 URL 提取日期 (這是最保險的日期來源)
        date_match = re.search(r"(\d{4}-\d{2}-\d{2})", latest_link)
        if not date_match:
            print("❌ 無法從網址提取日期"); return False
        formatted_date = date_match.group(1).replace("-", "/")
        
        # 4. 精確定位號碼球 (關鍵改動)
        balls = []
        # 在詳情頁中，號碼通常被包在 class 包含 'balls' 的 <ul> 或 <div class="draw-results"> 內
        # 我們先找到這個特定的結果容器，避免抓到網頁頂部的標題數字
        result_container = detail_soup.find(['ul', 'div'], class_=re.compile(r'balls|result-box|draw-results', re.I))
        
        if not result_container:
            # 如果找不到特定容器，則搜尋頁面中具有 'ball' 類別的元素
            ball_elements = detail_soup.find_all(['li', 'span'], class_=re.compile(r'ball|no-', re.I))
        else:
            ball_elements = result_container.find_all(['li', 'span'], class_=re.compile(r'ball|no-', re.I))
            
        for el in ball_elements:
            txt = el.get_text(strip=True)
            if txt.isdigit():
                val = int(txt)
                if 1 <= val <= 49 and val not in balls:
                    balls.append(val)
        
        # 只要前 7 個球 (6個正獎 + 1個特別獎)
        if len(balls) >= 7:
            main_nums = balls[:6]
            extra_num = balls[6]
            
            # 5. 抓取獎金資訊 (選填)
            full_text = detail_soup.get_text()
            prize_match = re.search(r'1st.*?HK\$?([\d,]+)', full_text, re.I | re.DOTALL)
            div1_prize = prize_match.group(1).replace(',', '') if prize_match else "0"
            
            print(f"✅ 成功抓取！日期: {formatted_date}")
            print(f"✅ 號碼: {main_nums} + 特別號: {extra_num}")

            # 6. 更新 CSV
            csv_path = 'marksix.csv'
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                df['date'] = df['date'].astype(str)
            else:
                df = pd.DataFrame(columns=['date', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'extra', 'div1_prize', 'url'])

            # 如果這一天已經有資料了，我們先刪除舊的錯誤資料再寫入新的
            if formatted_date in df['date'].values:
                print(f"ℹ️ 日期 {formatted_date} 已有紀錄，正在執行覆蓋更新...")
                df = df[df['date'] != formatted_date]

            new_data = {
                'date': formatted_date,
                'n1': main_nums[0], 'n2': main_nums[1], 'n3': main_nums[2],
                'n4': main_nums[3], 'n5': main_nums[4], 'n6': main_nums[5],
                'extra': extra_num, 'div1_prize': div1_prize, 'div1_winners': 0, 'url': latest_link
            }
            
            df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
            # 重新排序確保日期正確
            df['date_dt'] = pd.to_datetime(df['date'], format='%Y/%m/%d')
            df = df.sort_values('date_dt').drop(columns=['date_dt'])
            df.to_csv(csv_path, index=False)
            
            print("🚀 數據已完美存入 marksix.csv")
            return True
        else:
            print(f"❌ 解析失敗，只抓到 {len(balls)} 個號碼。")
            
    except Exception as e:
        print(f"❌ 發生錯誤: {e}")
    return False

if __name__ == "__main__":
    fetch_latest_marksix()
