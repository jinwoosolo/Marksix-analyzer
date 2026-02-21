import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import os
import time
from urllib.parse import urljoin
from datetime import datetime

# 配置
BASE_URL = "https://www.lottery.hk"
RESULTS_LIST_URL = "https://www.lottery.hk/en/mark-six/results"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def fetch_latest_marksix():
    """
    採用詳情頁抓取策略：
    1. 先到列表頁找出最新一期的網址
    2. 點進去詳情頁抓取唯一的數據
    """
    try:
        print(f"🚀 正在存取列表頁: {RESULTS_LIST_URL}")
        resp = requests.get(RESULTS_LIST_URL, headers=headers, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        
        # 1. 尋找最新一期的詳細連結 (例如 /en/mark-six/results/2026-02-21)
        latest_url = None
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if re.search(r"/mark-six/results/20\d{2}-\d{2}-\d{2}", href):
                latest_url = urljoin(BASE_URL, href)
                break
        
        if not latest_url:
            print("❌ 找不到最新一期的詳細連結"); return False
            
        print(f"🔎 進入詳情頁抓取: {latest_url}")
        
        # 2. 存取詳情頁
        time.sleep(1) # 稍微延遲避免被封
        detail_resp = requests.get(latest_url, headers=headers, timeout=20)
        detail_resp.raise_for_status()
        detail_soup = BeautifulSoup(detail_resp.text, "lxml")
        
        # 3. 從 URL 提取日期 (最準確)
        date_match = re.search(r"(\d{4}-\d{2}-\d{2})", latest_url)
        if not date_match:
            print("❌ 無法從 URL 提取日期"); return False
            
        formatted_date = date_match.group(1).replace("-", "/")
        
        # 4. 抓取號碼 (只從詳情頁的結果區域抓取)
        # 詳情頁通常有一個專門放號碼的容器
        balls = []
        # 鎖定詳情頁中的結果區塊 (通常是 .result-numbers 或類似容器)
        target_container = detail_soup.find(['div', 'ul'], class_=re.compile(r'result|ball|number', re.I))
        if not target_container:
            target_container = detail_soup
            
        # 尋找所有 1-49 的數字球
        for el in target_container.find_all(['li', 'span', 'div'], class_=re.compile(r'ball|no-|num', re.I)):
            txt = el.get_text(strip=True)
            if txt.isdigit():
                val = int(txt)
                if 1 <= val <= 49 and val not in balls:
                    balls.append(val)
        
        # 如果標籤法失敗，使用 Regex 掃描詳情頁 (詳情頁雜訊較少)
        if len(balls) < 7:
            # 詳情頁中通常會有一串連續的數字
            all_text = detail_soup.get_text(separator=' ', strip=True)
            # 尋找 7 個一組的號碼
            nums_match = re.findall(r'\b\d{1,2}\b', all_text)
            balls = [int(n) for n in nums_match if 1 <= int(n) <= 49][:7]

        if len(balls) >= 7:
            main_nums = balls[:6]
            extra_num = balls[6]
            
            print(f"✅ 成功偵測 - 日期: {formatted_date}")
            print(f"✅ 成功偵測 - 號碼: {main_nums} + {extra_num}")

            csv_path = 'marksix.csv'
            df = pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame()

            # 轉為字串比較
            if df.empty or formatted_date not in df['date'].astype(str).values:
                new_data = {
                    'date': formatted_date,
                    'n1': main_nums[0], 'n2': main_nums[1], 'n3': main_nums[2],
                    'n4': main_nums[3], 'n5': main_nums[4], 'n6': main_nums[5],
                    'extra': extra_num, 'div1_prize': 'TBA', 'div1_winners': 0, 
                    'url': latest_url
                }
                df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
                # 排序
                df['dt'] = pd.to_datetime(df['date'], format='%Y/%m/%d')
                df = df.sort_values('dt', ascending=True).drop(columns=['dt'])
                df.to_csv(csv_path, index=False)
                print(f"🎉 數據已成功更新至 marksix.csv: {formatted_date}")
                return True
            else:
                print(f"ℹ️ 日期 {formatted_date} 的數據已存在，略過更新。")
        else:
            print(f"❌ 詳情頁號碼偵測不足: {balls}")
            
    except Exception as e:
        print(f"❌ 發生異常錯誤: {e}")
    return False

if __name__ == "__main__":
    fetch_latest_marksix()
