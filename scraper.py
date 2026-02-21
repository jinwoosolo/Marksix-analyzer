import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from datetime import datetime
import os

def fetch_latest_marksix():
    """超強版：支持金多寶特殊頁面結構抓取"""
    url = "https://bet.hkjc.com/en/marksix/results"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9"
    }

    try:
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()
        # 強制使用 utf-8 解碼避免亂碼
        response.encoding = 'utf-8'
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # --- 策略 A: 尋找日期 ---
        text_content = soup.get_text(separator=' ', strip=True)
        # 尋找 DD/MM/YYYY 格式
        date_match = re.search(r'(\d{1,2}/\d{1,2}/20\d{2})', text_content)
        
        # --- 策略 B: 抓取號碼 ---
        balls = []
        
        # 1. 嘗試從圖片檔名提取 (no_01.gif)
        for img in soup.find_all('img'):
            src = img.get('src', '')
            match = re.search(r'no_(\d+)\.gif', src)
            if match:
                val = int(match.group(1))
                # 避免重複抓取 (有些頁面會顯示兩次)
                if len(balls) < 7:
                    balls.append(val)

        # 2. 如果圖片法失效，嘗試從 class 包含 'ball' 的元素提取文本
        if len(balls) < 7:
            ball_elements = soup.find_all(class_=re.compile(r'ball|no_', re.I))
            temp_balls = []
            for el in ball_elements:
                val_text = el.get_text(strip=True)
                if val_text.isdigit():
                    temp_balls.append(int(val_text))
            if len(temp_balls) >= 7:
                balls = temp_balls[:7]

        # 3. 如果還是不行，使用正則表達式在純文本中尋找 6+1 模式
        if len(balls) < 7:
            # 匹配 7 個數字，中間可能有空格或 + 號
            nums_match = re.search(r'(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*[\+\&]?\s*(\d+)', text_content)
            if nums_match:
                balls = [int(nums_match.group(i)) for i in range(1, 8)]

        if date_match and len(balls) >= 7:
            raw_date = date_match.group(1)
            formatted_date = datetime.strptime(raw_date, '%d/%m/%Y').strftime('%Y/%m/%d')
            main_nums = balls[:6]
            extra_num = balls[6]
            
            print(f"🔍 偵測日期: {formatted_date}")
            print(f"🔍 偵測號碼: {main_nums} + {extra_num}")

            csv_path = 'marksix.csv'
            # 讀取現有數據
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                # 確保日期列是字串以便比對
                df['date'] = df['date'].astype(str)
            else:
                df = pd.DataFrame(columns=['date', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'extra'])

            if formatted_date not in df['date'].values:
                new_row = {
                    'date': formatted_date,
                    'n1': main_nums[0], 'n2': main_nums[1], 'n3': main_nums[2],
                    'n4': main_nums[3], 'n5': main_nums[4], 'n6': main_nums[5],
                    'extra': extra_num,
                    'div1_prize': 'TBA', 'div1_winners': 0, 'url': url
                }
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                # 排序
                df['dt'] = pd.to_datetime(df['date'], format='%Y/%m/%d')
                df = df.sort_values('dt').drop(columns=['dt'])
                df.to_csv(csv_path, index=False)
                print(f"✅ CSV 更新完成")
                return True
            else:
                print(f"ℹ️ {formatted_date} 的數據已存在")
        else:
            print(f"❌ 數據識別失敗。日期找到: {date_match is not None}, 號碼球數量: {len(balls)}")
            # 印出部分文本內容以便除錯
            print(f"頁面文字片段: {text_content[:200]}")
            
    except Exception as e:
        print(f"❌ 程式錯誤: {e}")
    
    return False

if __name__ == "__main__":
    fetch_latest_marksix()
