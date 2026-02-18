# scraper.py (每日抓 HKJC 最新)
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re

def add_latest_draw():
    """抓 HKJC 最新開彩"""
    url = "https://bet.hkjc.com/en/marksix/results"
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, 'html.parser')
    text = soup.get_text()
    
    # 最新一期 pattern
    date_match = re.search(r'(\d{1,2}/\d{1,2}/20\d{2})', text)
    nums_match = re.search(r'(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*\+\s*(\d+)', text)
    
    if date_match and nums_match:
        date = date_match.group(1)
        nums = [int(nums_match.group(i)) for i in range(1,8)]
        
        new_row = pd.DataFrame([{
            'date': date,
            'n1': nums[0], 'n2': nums[1], 'n3': nums[2],
            'n4': nums[3], 'n5': nums[4], 'n6': nums[5],
            'extra': nums[6],
            'div1_prize': 'Live',
            'div1_winners': 0,
            'url': url
        }])
        
        try:
            df = pd.read_csv('marksix.csv')
            if date not in df['date'].values:
                df = pd.concat([df, new_row], ignore_index=True)
                df.to_csv('marksix.csv', index=False)
                print(f"✅ Added {date}")
                return True
        except FileNotFoundError:
            new_row.to_csv('marksix.csv', index=False)
    
    print("No new draw found")
    return False

if __name__ == "__main__":
    add_latest_draw()
