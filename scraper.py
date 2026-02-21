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
    """採用廣域掃描邏輯，抓取頁面上第一組有效的六合彩結果"""
    try:
        print(f"🚀 開始抓取: {TARGET_URL}")
        resp = requests.get(TARGET_URL, headers=headers, timeout=20)
        resp.raise_for_status()
        # 嘗試使用 lxml，如果環境沒有則退回 html.parser
        try:
            soup = BeautifulSoup(resp.text, "lxml")
        except:
            soup = BeautifulSoup(resp.text, "html.parser")
        
        # --- 步驟 1: 尋找日期 ---
        # 遍歷頁面文本尋找第一個符合 DD Month YYYY 格式的日期
        full_text = soup.get_text(separator=' ', strip=True)
        date_pattern = r'(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{4})'
        date_match = re.search(date_pattern, full_text)
        
        if not date_match:
            print("❌ 在頁面上找不到任何日期數據"); return False
            
        day, month_str, year = date_match.groups()
        months_map = {
            'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05', 'Jun': '06',
            'Jul': '07', 'Aug': '08', 'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12',
            'January': '01', 'February': '02', 'March': '03', 'April': '04', 'May': '05', 'June': '06',
            'July': '07', 'August': '08', 'September': '09', 'October': '10', 'November': '11', 'December': '12'
        }
        m_code = months_map.get(month_str[:3].capitalize(), '01')
        formatted_date = f"{year}/{m_code}/{day.zfill(2)}"
        
        # --- 步驟 2: 抓取號碼 ---
        # 策略：尋找所有標籤，只要內容是 1-49 的數字且具有「球」的特徵
        balls = []
        # 尋找所有 <li>, <span>, <div>
        for el in soup.find_all(['li', 'span', 'div']):
            cls = str(el.get('class', ''))
            txt = el.get_text(strip=True)
            if txt.isdigit():
                val = int(txt)
                # 判斷是否為號碼球：內容 1-49 且 class 包含 ball 或 no- 或 num
                if 1 <= val <= 49 and re.search(r'ball|no-|num|result', cls, re.I):
                    if len(balls) < 7: # 只要前 7 個
                        # 避免抓到日期中的數字
                        if val == int(day) and "date" in cls.lower():
                            continue
                        balls.append(val)
        
        # 如果標籤掃描失敗，改用正則在日期附近的文本提取數字
        if len(balls) < 7:
            # 找到日期後面的文本片段
            after_date_text = full_text[date_match.end():date_match.end()+200]
            nums_in_text = re.findall(r'\b\d{1,2}\b', after_date_text)
            balls = [int(n) for n in nums_in_text if 1 <= int(n) <= 49][:7]

        if len(balls) >= 7:
            main_nums = balls[:6]
            extra_num = balls[6]
            
            print(f"✅ 成功偵測 - 日期: {formatted_date}")
            print(f"✅ 成功偵測 - 號碼: {main_nums} + {extra_num}")

            csv_path = 'marksix.csv'
            df = pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame()

            # 轉為字串比較，避免重複紀錄
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
                print(f"🎉 已成功更新 CSV 檔案！")
                return True
            else:
                print(f"ℹ️ 日期 {formatted_date} 已存在，無需更新。")
        else:
            print(f"❌ 號碼偵測不足 (只抓到 {len(balls)} 個): {balls}")
            
    except Exception as e:
        print(f"❌ 發生異常錯誤: {e}")
    return False

if __name__ == "__main__":
    fetch_latest_marksix()
