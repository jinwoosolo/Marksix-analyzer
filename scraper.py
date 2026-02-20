import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from datetime import datetime

def add_latest_draw():
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    url = "https://bet.hkjc.com/en/marksix/results"
    
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(resp.text, 'html.parser')
        text = soup.get_text()
        
        # 尋找日期與號碼
        date_match = re.search(r'(\d{1,2}/\d{1,2}/20\d{2})', text)
        nums_match = re.search(r'(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*\+\s*(\d+)', text)
        
        if date_match and nums_match:
            # 統一存儲格式為 YYYY/MM/DD
            scraped_date = datetime.strptime(date_match.group(1), '%d/%m/%Y').strftime('%Y/%m/%d')
            nums = [int(nums_match.group(i)) for i in range(1, 8)]
            
            df = pd.read_csv('marksix.csv')
            if scraped_date not in df['date'].values:
                new_row = pd.DataFrame([{
                    'date': scraped_date,
                    'n1': nums[0], 'n2': nums[1], 'n3': nums[2],
                    'n4': nums[3], 'n5': nums[4], 'n6': nums[5],
                    'extra': nums[6],
                    'div1_prize': 'Pending',
                    'div1_winners': 0,
                    'url': url
                }])
                df = pd.concat([df, new_row], ignore_index=True)
                df.to_csv('marksix.csv', index=False)
                print(f"✅ 已更新數據: {scraped_date}")
                return True
            else:
                print("ℹ️ 數據已是最新")
    except Exception as e:
        print(f"❌ 抓取錯誤: {e}")
    return False

if __name__ == "__main__":
    add_latest_draw()
