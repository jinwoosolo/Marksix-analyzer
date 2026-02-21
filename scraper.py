import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import os
import time
from datetime import datetime

# 配置
TARGET_URL = "https://www.lottery.hk/en/mark-six/results"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def fetch_latest_marksix():
    """精確抓取 lottery.hk 最頂端的第一個結果"""
    try:
        print(f"🚀 開始抓取: {TARGET_URL}")
        resp = requests.get(TARGET_URL, headers=headers, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        
        # 1. 鎖定第一個結果容器 (這是關鍵，避免抓到下方的歷史紀錄)
        # lottery.hk 的最新結果通常在第一個 .result-box 或 table 的第一行
        latest_box = soup.find('div', class_='result-box')
        if not latest_box:
            # 備選方案：抓取表格的第一列
            latest_box = soup.find('tr', class_='result-row')
        
        if not latest_box:
            # 最後手段：掃描第一個包含號碼球的區域
            latest_box = soup.find('ul', class_='results-list')
            if latest_box:
                latest_box = latest_box.find_parent('div')

        if not latest_box:
            print("❌ 找不到結果容器"); return False

        # 2. 抓取日期 (精確從該容器抓取)
        date_text = latest_box.get_text(separator=' ', strip=True)
        # 尋找格式如: 21 February 2026
        date_match = re.search(r'(\d{1,2})\s+([A-Z][a-z]+)\s+(\d{4})', date_text)
        
        if not date_match:
            print("❌ 找不到日期數據"); return False
            
        day, month_str, year = date_match.groups()
        months_map = {
            'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05', 'Jun': '06',
            'Jul': '07', 'Aug': '08', 'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12',
            'January': '01', 'February': '02', 'March': '03', 'April': '04', 'May': '05', 'June': '06',
            'July': '07', 'August': '08', 'September': '09', 'October': '10', 'November': '11', 'December': '12'
        }
        m_code = months_map.get(month_str[:3].capitalize(), '01')
        formatted_date = f"{year}/{m_code}/{day.zfill(2)}"
        
        # 3. 抓取號碼 (只從該容器內找球)
        balls = []
        # 尋找容器內所有的號碼元素
        ball_elements = latest_box.find_all(['span', 'li', 'div'], class_=re.compile(r'ball|no-|num', re.I))
        
        for el in ball_elements:
            txt = el.get_text(strip=True)
            if txt.isdigit():
                val = int(txt)
                if 1 <= val <= 49 and val not in balls:
                    balls.append(val)
        
        # 如果標籤法失敗，再用正則分開提取
        if len(balls) < 7:
            # 透過尋找獨立的 1-2 位數字來避免黏在一起
            potential = re.findall(r'\b\d{1,2}\b', date_text)
            # 排除掉日期中的數字 (year, day)
            filtered = [int(n) for n in potential if 1 <= int(n) <= 49 and n != day and n != year]
            balls = filtered[:7]

        if len(balls) >= 7:
            main_nums = balls[:6]
            extra_num = balls[6]
            
            print(f"✅ 偵測到日期: {formatted_date}")
            print(f"✅ 偵測到號碼: {main_nums} + {extra_num}")

            csv_path = 'marksix.csv'
            df = pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame()

            if formatted_date not in df['date'].astype(str).values:
                new_data = {
                    'date': formatted_date,
                    'n1': main_nums[0], 'n2': main_nums[1], 'n3': main_nums[2],
                    'n4': main_nums[3], 'n5': main_nums[4], 'n6': main_nums[5],
                    'extra': extra_num, 'div1_prize': 'TBA', 'div1_winners': 0, 
                    'url': TARGET_URL
                }
                df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
                df['dt'] = pd.to_datetime(df['date'], format='%Y/%m/%d')
                df = df.sort_values('dt').drop(columns=['dt'])
                df.to_csv(csv_path, index=False)
                print(f"🚀 CSV 已更新: {formatted_date}")
                return True
            else:
                print(f"ℹ️ {formatted_date} 數據已存在。")
        else:
            print(f"❌ 號碼不足 (抓到 {len(balls)} 個): {balls}")
            
    except Exception as e:
        print(f"❌ 發生錯誤: {e}")
    return False

if __name__ == "__main__":
    fetch_latest_marksix()
