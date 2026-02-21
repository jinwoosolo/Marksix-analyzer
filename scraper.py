import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from datetime import datetime
import os

def fetch_latest_marksix():
    """強化版抓取器：針對 lottery.hk 進行多路徑掃描"""
    url = "https://www.lottery.hk/en/mark-six/results"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
    }

    try:
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 1. 尋找數據容器 (嘗試多種可能的容器)
        container = soup.find('div', class_=re.compile(r'result|latest|content', re.I))
        if not container:
            container = soup
            
        full_text = container.get_text(separator=' ', strip=True)
        
        # 2. 抓取日期 (更寬鬆的匹配模式)
        # 匹配: 21 Feb 2026 或 21 February 2026
        date_pattern = r'(\d{1,2})\s+([A-Z][a-z]+)\s+(\d{4})'
        date_match = re.search(date_pattern, full_text)
        
        # 3. 抓取號碼 (尋找所有球狀或列表元素中的數字)
        balls = []
        # 遍歷所有 span, li, div，尋找可能是號碼球的元素
        for el in container.find_all(['span', 'li', 'div', 'td']):
            cls = "".join(el.get('class', []))
            txt = el.get_text(strip=True)
            # 如果 class 包含 ball 或數字內容在 1-49 之間
            if (re.search(r'ball|no-|num', cls, re.I) or 'ball' in el.get('id', '')) and txt.isdigit():
                val = int(txt)
                if 1 <= val <= 49 and val not in balls:
                    balls.append(val)
        
        # 如果還是抓不到，強行從文本中提取前 7 個獨立數字
        if len(balls) < 7:
            # 尋找 7 個 1-2 位數的組合
            potential_nums = re.findall(r'\b\d{1,2}\b', full_text)
            balls = [int(n) for n in potential_nums if 1 <= int(n) <= 49][:7]

        if date_match and len(balls) >= 7:
            day, month_str, year = date_match.groups()
            
            # 日期標準化處理
            months_map = {
                'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05', 'Jun': '06',
                'Jul': '07', 'Aug': '08', 'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12',
                'January': '01', 'February': '02', 'March': '03', 'April': '04', 'June': '06',
                'July': '07', 'August': '08', 'September': '09', 'October': '10', 'November': '11', 'December': '12'
            }
            m_code = months_map.get(month_str[:3].capitalize(), '01')
            formatted_date = f"{year}/{m_code}/{day.zfill(2)}"

            # lottery.hk 通常最後一個是特別號
            main_nums = sorted(balls[:6])
            extra_num = balls[6]
            
            print(f"✅ 偵測到日期: {formatted_date}")
            print(f"✅ 偵測到號碼: {main_nums} + {extra_num}")

            csv_path = 'marksix.csv'
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
            else:
                df = pd.DataFrame(columns=['date', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'extra'])

            if formatted_date not in df['date'].astype(str).values:
                new_row = {
                    'date': formatted_date,
                    'n1': main_nums[0], 'n2': main_nums[1], 'n3': main_nums[2],
                    'n4': main_nums[3], 'n5': main_nums[4], 'n6': main_nums[5],
                    'extra': extra_num, 'div1_prize': 'TBA', 'div1_winners': 0, 'url': url
                }
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                # 日期排序
                df['date_dt'] = pd.to_datetime(df['date'], format='%Y/%m/%d')
                df = df.sort_values('date_dt').drop(columns=['date_dt'])
                df.to_csv(csv_path, index=False)
                print(f"🚀 CSV 已更新並保存。")
                return True
            else:
                print(f"ℹ️ {formatted_date} 的數據已存在。")
        else:
            print(f"❌ 數據識別不完全。日期: {date_match is not None}, 球數: {len(balls)}")
            print(f"文本片段: {full_text[:300]}")
            
    except Exception as e:
        print(f"❌ 發生錯誤: {e}")
    
    return False

if __name__ == "__main__":
    fetch_latest_marksix()
