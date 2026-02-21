import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from datetime import datetime
import os

def fetch_latest_marksix():
    """優化版：精確抓取馬會最新一期號碼"""
    # 使用馬會的結果摘要頁面，這通常對爬蟲更友好
    url = "https://bet.hkjc.com/en/marksix/results"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9"
    }

    try:
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 1. 抓取日期
        # 馬會日期通常在 class="draw_date" 或包含在特定的 td/div 中
        text_content = soup.get_text(separator=' ', strip=True)
        date_match = re.search(r'(\d{1,2}/\d{1,2}/20\d{2})', text_content)
        
        # 2. 抓取號碼球
        # 尋找所有像號碼球的元素 (通常有特殊的圖片路徑或 class)
        # 我們直接從 img 標籤的 alt 或 src 中提取號碼
        balls = []
        for img in soup.find_all('img'):
            src = img.get('src', '')
            # 號碼球圖片通常命名為 no_01.gif, no_02.gif ...
            match = re.search(r'no_(\d+)\.gif', src)
            if match:
                balls.append(int(match.group(1)))
        
        # 如果透過圖片抓不到，嘗試正則匹配文本中的 6+1 模式
        if len(balls) < 7:
            nums_match = re.search(r'(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*\+\s*(\d+)', text_content)
            if nums_match:
                balls = [int(nums_match.group(i)) for i in range(1, 8)]

        if date_match and len(balls) >= 7:
            raw_date = date_match.group(1)
            formatted_date = datetime.strptime(raw_date, '%d/%m/%Y').strftime('%Y/%m/%d')
            # 只需要前 6 個正獎和第 7 個特別獎
            main_nums = balls[:6]
            extra_num = balls[6]
            
            print(f"🔍 偵測到日期: {formatted_date}")
            print(f"🔍 偵測到號碼: {main_nums} + {extra_num}")

            csv_path = 'marksix.csv'
            df = pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame()

            if formatted_date not in df['date'].values.astype(str):
                new_row = {
                    'date': formatted_date,
                    'n1': main_nums[0], 'n2': main_nums[1], 'n3': main_nums[2],
                    'n4': main_nums[3], 'n5': main_nums[4], 'n6': main_nums[5],
                    'extra': extra_num,
                    'div1_prize': 'TBA', 'div1_winners': 0, 'url': url
                }
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                df['date_dt'] = pd.to_datetime(df['date'], format='%Y/%m/%d')
                df = df.sort_values('date_dt', ascending=True).drop(columns=['date_dt'])
                df.to_csv(csv_path, index=False)
                print(f"✅ 成功更新 CSV 數據")
                return True
            else:
                print(f"ℹ️ 數據已存在，無需更新")
        else:
            print(f"❌ 抓取失敗。找到日期: {date_match is not None}, 找到號碼球數量: {len(balls)}")
            
    except Exception as e:
        print(f"❌ 執行錯誤: {e}")
    
    return False

if __name__ == "__main__":
    fetch_latest_marksix()
