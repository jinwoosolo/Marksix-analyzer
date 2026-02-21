import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from datetime import datetime
import os

def fetch_latest_marksix():
    """從 lottery.hk 抓取最新一期六合彩結果"""
    # 切換至 lottery.hk 來源
    url = "https://www.lottery.hk/en/mark-six/results"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    try:
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 1. 抓取最新一期的容器
        # lottery.hk 的結果通常放在 class="results-table" 或第一個結果區塊
        latest_result_box = soup.find('div', class_='result-box')
        if not latest_result_box:
            latest_result_box = soup # 如果找不到特定容器則搜尋全頁
            
        full_text = latest_result_box.get_text(separator=' ', strip=True)
        
        # 2. 抓取日期
        # lottery.hk 常用格式如: "Saturday 21 February 2026" 或 "21 Feb 2026"
        # 我們搜尋包含日、月、年的字串
        date_pattern = r'(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{4})'
        date_match = re.search(date_pattern, full_text)
        
        # 3. 抓取號碼球
        # lottery.hk 的號碼通常放在 class 為 ball 的 span 或 li 標籤中
        balls = []
        ball_elements = latest_result_box.find_all(class_=re.compile(r'ball|no-', re.I))
        for el in ball_elements:
            val_text = el.get_text(strip=True)
            if val_text.isdigit():
                balls.append(int(val_text))
        
        # 如果標籤法抓不到，嘗試從文本中抓取連續的數字
        if len(balls) < 7:
            # 尋找 7 個一組的數字
            nums_match = re.findall(r'\b\d{1,2}\b', full_text)
            temp_balls = [int(n) for n in nums_match if 1 <= int(n) <= 49]
            if len(temp_balls) >= 7:
                balls = temp_balls[:7]

        if date_match and len(balls) >= 7:
            day = date_match.group(1)
            month_str = date_match.group(2)
            year = date_match.group(3)
            
            # 轉換日期格式為 YYYY/MM/DD
            try:
                # 先嘗試完整月份名，再嘗試縮寫
                fmt = '%d %B %Y' if len(month_str) > 3 else '%d %b %Y'
                dt_obj = datetime.strptime(f"{day} {month_str} {year}", fmt)
                formatted_date = dt_obj.strftime('%Y/%m/%d')
            except:
                # 備用手動映射
                months = {
                    'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05', 'Jun': '06',
                    'Jul': '07', 'Aug': '08', 'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
                }
                m_code = months.get(month_str[:3], '01')
                formatted_date = f"{year}/{m_code}/{day.zfill(2)}"

            main_nums = balls[:6]
            extra_num = balls[6]
            
            print(f"✅ 來源: lottery.hk")
            print(f"✅ 偵測到日期: {formatted_date}")
            print(f"✅ 偵測到號碼: {main_nums} + {extra_num}")

            csv_path = 'marksix.csv'
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
            else:
                df = pd.DataFrame(columns=['date', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'extra'])

            # 轉為字串比較，避免重複
            if formatted_date not in df['date'].astype(str).values:
                new_row = {
                    'date': formatted_date,
                    'n1': main_nums[0], 'n2': main_nums[1], 'n3': main_nums[2],
                    'n4': main_nums[3], 'n5': main_nums[4], 'n6': main_nums[5],
                    'extra': extra_num,
                    'div1_prize': 'TBA', 'div1_winners': 0, 'url': url
                }
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                # 按日期排序並覆寫
                df['date_dt'] = pd.to_datetime(df['date'], format='%Y/%m/%d')
                df = df.sort_values('date_dt').drop(columns=['date_dt'])
                df.to_csv(csv_path, index=False)
                print("🚀 數據已成功同步至 marksix.csv")
                return True
            else:
                print(f"ℹ️ {formatted_date} 的數據已存在。")
        else:
            print(f"❌ 無法識別 lottery.hk 的數據。日期找到: {date_match is not None}, 球數: {len(balls)}")
            
    except Exception as e:
        print(f"❌ 執行出錯: {e}")
    
    return False

if __name__ == "__main__":
    fetch_latest_marksix()
