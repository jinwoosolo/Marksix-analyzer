# scraper.py（放 repo root）
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from datetime import datetime

def scrape_latest_from_hkjc():
    """從 HKJC 官網抓最新開彩"""
    url = "https://bet.hkjc.com/en/marksix/results"
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, 'html.parser')
    
    # 最新一期通常喺最顯眼位
    text = soup.get_text()
    
    # 日期
    date_match = re.search(r'(\d{1,2}/\d{1,2}/\d{4})', text)
    date = date_match.group(1) if date_match else datetime.now().strftime('%d/%m/%Y')
    
    # 號碼
    nums_match = re.search(r'(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*\+\s*(\d+)', text)
    if nums_match:
        nums = [int(nums_match.group(i)) for i in range(1,8)]
        new_row = {
            'date': date,
            'n1': nums[0], 'n2': nums[1], 'n3': nums[2],
            'n4': nums[3], 'n5': nums[4], 'n6': nums[5],
            'extra': nums[6],
            'div1_prize': 'TBD',  # 開彩後手動或再抓
            'div1_winners': 0
        }
        
        # Append 到現有 CSV
        try:
            existing = pd.read_csv('marksix.csv')
            updated = pd.concat([existing, pd.DataFrame([new_row])], ignore_index=True)
            updated.to_csv('marksix.csv', index=False)
            print(f"✅ Added {date}")
            return True
        except:
            pd.DataFrame([new_row]).to_csv('marksix.csv', index=False)
            print("✅ Created new CSV")
    
    return False

if __name__ == "__main__":
    scrape_latest_from_hkjc()
