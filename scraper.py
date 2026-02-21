import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from datetime import datetime
import os

def fetch_latest_marksix():
    """強化版抓取器：支持金多寶頁面與圖片解析"""
    url = "https://bet.hkjc.com/en/marksix/results"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    try:
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 提取純文本內容
        full_text = soup.get_text(separator=' ', strip=True)
        
        # 1. 抓取日期 (格式: DD/MM/YYYY)
        date_match = re.search(r'(\d{1,2}/\d{1,2}/20\d{2})', full_text)
        
        # 2. 抓取號碼球 (優先從圖片 src 抓取 no_xx.gif)
        balls = []
        for img in soup.find_all('img'):
            src = img.get('src', '')
            match = re.search(r'no_(\d+)\.gif', src)
            if match:
                val = int(match.group(1))
                if len(balls) < 7: # 只要前 7 個球
                    balls.append(val)

        # 備用方案：如果圖片抓不到，搜尋文本中的號碼規律
        if len(balls) < 7:
            # 搜尋類似 2 18 34 35 37 49 + 33 的模式
            nums_match = re.search(r'(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*[\+\&]\s*(\d+)', full_text)
            if nums_match:
                balls = [int(nums_match.group(i)) for i in range(1, 8)]

        if date_match and len(balls) >= 7:
            raw_date = date_match.group(1)
            formatted_date = datetime.strptime(raw_date, '%d/%m/%Y').strftime('%Y/%m/%d')
            main_nums = balls[:6]
            extra_num = balls[6]
            
            print(f"✅ 偵測到日期: {formatted_date}")
            print(f"✅ 偵測到號碼: {main_nums} + {extra_num}")

            csv_path = 'marksix.csv'
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
            else:
                df = pd.DataFrame(columns=['date', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'extra'])

            # 轉為字串比較
            if formatted_date not in df['date'].astype(str).values:
                new_row = {
                    'date': formatted_date,
                    'n1': main_nums[0], 'n2': main_nums[1], 'n3': main_nums[2],
                    'n4': main_nums[3], 'n5': main_nums[4], 'n6': main_nums[5],
                    'extra': extra_num,
                    'div1_prize': 'TBA', 'div1_winners': 0, 'url': url
                }
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                # 排序並保存
                df['date_dt'] = pd.to_datetime(df['date'], format='%Y/%m/%d')
                df = df.sort_values('date_dt').drop(columns=['date_dt'])
                df.to_csv(csv_path, index=False)
                print("🚀 數據已成功寫入 CSV")
                return True
            else:
                print(f"ℹ️ {formatted_date} 的數據已存在，無需更新。")
        else:
            print(f"❌ 抓取數據不完整。日期: {date_match is not None}, 球數: {len(balls)}")
            
    except Exception as e:
        print(f"❌ 執行出錯: {e}")
    
    return False

if __name__ == "__main__":
    fetch_latest_marksix()
