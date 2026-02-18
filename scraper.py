# scraper.py 最終版
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re

def scrape_hkjc_latest():
    """HKJC 官網最新（2026/2）"""
    url = "https://bet.hkjc.com/en/marksix/results"
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, 'html.parser')
    text = soup.get_text()
    
    # 最新日期
    date_m = re.search(r'(\d{1,2}/\d{1,2}/\d{4})', text)
    date = date_m.group(1) if date_m else None
    
    # 號碼
    nums_m = re.search(r'(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*\+\s*(\d+)', text)
    
    if nums_m and date:
        new_row = {
            'date': date,
            *[f'n{i}': int(nums_m.group(i)) for i in range(1,7)],
            'extra': int(nums_m.group(7)),
            'div1_prize': 'Live',  # 開彩後更新
            'div1_winners': 0
        }
        
        df = pd.read_csv('marksix.csv')
        if date not in df['date'].values:
            df = pd.concat([df, pd.DataFrame([new_row])])
            df.to_csv('marksix.csv', index=False)
            print(f"✅ Added {date}: {new_row['n1']}-{new_row['n2']}...")
    
    return True

if __name__ == "__main__":
    scrape_hkjc_latest()
