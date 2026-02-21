import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from datetime import datetime
import os

def fetch_latest_marksix():
    """從賽馬會抓取最新一期六合彩結果"""
    url = "https://bet.hkjc.com/en/marksix/results"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 獲取頁面文本內容進行正則匹配
        text = soup.get_text()
        
        # 匹配日期 (格式通常為 DD/MM/YYYY)
        date_match = re.search(r'(\d{1,2}/\d{1,2}/20\d{2})', text)
        # 匹配號碼 (通常是 6 個數字 + 1 個特別號碼)
        nums_match = re.search(r'(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*\+\s*(\d+)', text)
        
        if date_match and nums_match:
            raw_date = date_match.group(1)
            # 轉換為 app.py 使用的 YYYY/MM/DD 格式
            formatted_date = datetime.strptime(raw_date, '%d/%m/%Y').strftime('%Y/%m/%d')
            nums = [int(nums_match.group(i)) for i in range(1, 8)]
            
            # 讀取現有 CSV
            csv_path = 'marksix.csv'
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
            else:
                # 如果檔案不存在，建立一個基本的結構
                df = pd.DataFrame(columns=['date', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'extra', 'div1_prize', 'div1_winners', 'url'])

            # 檢查是否已經存在這期數據
            if formatted_date not in df['date'].values:
                new_row = {
                    'date': formatted_date,
                    'n1': nums[0], 'n2': nums[1], 'n3': nums[2],
                    'n4': nums[3], 'n5': nums[4], 'n6': nums[5],
                    'extra': nums[6],
                    'div1_prize': 'TBA',
                    'div1_winners': 0,
                    'url': url
                }
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                # 確保按日期排序
                df['date_dt'] = pd.to_datetime(df['date'], format='%Y/%m/%d')
                df = df.sort_values('date_dt', ascending=True).drop(columns=['date_dt'])
                df.to_csv(csv_path, index=False)
                print(f"✅ 成功更新數據：{formatted_date}")
                return True
            else:
                print(f"ℹ️ 數據已是最新：{formatted_date}")
        else:
            print("❌ 無法在頁面上找到日期或號碼數據")
            
    except Exception as e:
        print(f"❌ 抓取錯誤: {e}")
    
    return False

if __name__ == "__main__":
    fetch_latest_marksix()
