import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import os
from datetime import datetime

# 配置
TARGET_URL = "https://www.lottery.hk/en/mark-six/results"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
}

def fetch_latest_marksix():
    """精準版：只抓取網頁中第一個出現的開獎結果區塊"""
    try:
        print(f"🚀 開始抓取: {TARGET_URL}")
        resp = requests.get(TARGET_URL, headers=headers, timeout=20)
        resp.raise_for_status()
        
        # 優先使用 lxml 提升解析穩定度
        try:
            soup = BeautifulSoup(resp.text, "lxml")
        except:
            soup = BeautifulSoup(resp.text, "html.parser")
        
        # --- 步驟 1: 鎖定第一個結果容器 ---
        # lottery.hk 的結構中，最新結果通常在第一個 .result-box 或第一個 .result-row
        container = soup.find('div', class_='result-box')
        if not container:
            container = soup.find('tr', class_='result-row')
        
        # 如果找不到特定容器，我們縮小範圍至包含 "ball" 的第一個父級區塊
        if not container:
            first_ball = soup.find(class_=re.compile(r'ball|no-', re.I))
            if first_ball:
                container = first_ball.find_parent(['div', 'tr', 'li'])

        if not container:
            print("❌ 找不到最新的結果區塊"); return False

        # --- 步驟 2: 只在該容器內搜尋日期 ---
        # 這樣就不會抓到網頁下方或側邊欄的舊日期
        container_text = container.get_text(separator=' ', strip=True)
        date_pattern = r'(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{4})'
        date_match = re.search(date_pattern, container_text)
        
        if not date_match:
            # 如果容器內沒找到，嘗試在容器之前的標題找 (有時日期在區塊上方)
            prev_content = container.find_previous_sibling()
            if prev_content:
                date_match = re.search(date_pattern, prev_content.get_text())

        if not date_match:
            print("❌ 在目標區塊內找不到日期"); return False
            
        day, month_str, year = date_match.groups()
        months_map = {
            'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05', 'Jun': '06',
            'Jul': '07', 'Aug': '08', 'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12',
            'January': '01', 'February': '02', 'March': '03', 'April': '04', 'May': '05', 'June': '06',
            'July': '07', 'August': '08', 'September': '09', 'October': '10', 'November': '11', 'December': '12'
        }
        m_code = months_map.get(month_str[:3].capitalize(), '01')
        formatted_date = f"{year}/{m_code}/{day.zfill(2)}"
        
        # --- 步驟 3: 抓取號碼 ---
        balls = []
        # 只在該容器內找球
        ball_elements = container.find_all(['span', 'li', 'div'], class_=re.compile(r'ball|no-|num|result', re.I))
        
        for el in ball_elements:
            txt = el.get_text(strip=True)
            if txt.isdigit():
                val = int(txt)
                if 1 <= val <= 49 and val not in balls:
                    balls.append(val)
        
        # 如果標籤掃描不到，使用 Regex 在該容器文本中分開抓取
        if len(balls) < 7:
            # 尋找所有 1-2 位數字，並排除日期中的數字
            nums_in_text = re.findall(r'\b\d{1,2}\b', container_text)
            balls = []
            for n in nums_in_text:
                n_int = int(n)
                if 1 <= n_int <= 49 and n != day and n != year[2:]:
                    if n_int not in balls:
                        balls.append(n_int)
            balls = balls[:7]

        if len(balls) >= 7:
            # 前 6 個是正獎，第 7 個是特別號
            main_nums = balls[:6]
            extra_num = balls[6]
            
            print(f"✅ 成功偵測 - 日期: {formatted_date}")
            print(f"✅ 成功偵測 - 號碼: {main_nums} + {extra_num}")

            csv_path = 'marksix.csv'
            df = pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame()

            # 轉為字串比較，確保不會重複
            if df.empty or formatted_date not in df['date'].astype(str).values:
                new_data = {
                    'date': formatted_date,
                    'n1': main_nums[0], 'n2': main_nums[1], 'n3': main_nums[2],
                    'n4': main_nums[3], 'n5': main_nums[4], 'n6': main_nums[5],
                    'extra': extra_num, 'div1_prize': 'TBA', 'div1_winners': 0, 
                    'url': TARGET_URL
                }
                df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
                # 排序
                df['dt'] = pd.to_datetime(df['date'], format='%Y/%m/%d')
                df = df.sort_values('dt', ascending=True).drop(columns=['dt'])
                df.to_csv(csv_path, index=False)
                print(f"🎉 數據已更新至 marksix.csv")
                return True
            else:
                print(f"ℹ️ 日期 {formatted_date} 已在 CSV 中，略過更新。")
        else:
            print(f"❌ 號碼抓取不足 (抓到 {len(balls)} 個): {balls}")
            
    except Exception as e:
        print(f"❌ 發生異常錯誤: {e}")
    return False

if __name__ == "__main__":
    fetch_latest_marksix()
