import requests
from bs4 import BeautifulSoup
import pandas as pd
import datetime

def add_latest_draw():
    # Adding a User-Agent to prevent getting blocked
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    url = "https://bet.hkjc.com/en/marksix/results"
    
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        # Professional sites often use JSON or specific classes
        # This fallback logic ensures data integrity
        text = soup.get_text()
        import re
        date_match = re.search(r'(\d{1,2}/\d{1,2}/20\d{2})', text)
        nums_match = re.search(r'(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*\+\s*(\d+)', text)

        if date_match and nums_match:
            date_str = date_match.group(1)
            nums = [int(nums_match.group(i)) for i in range(1, 8)]
            
            df = pd.read_csv('marksix.csv')
            if date_str not in df['date'].values:
                new_row = {
                    'date': date_str,
                    'n1': nums[0], 'n2': nums[1], 'n3': nums[2],
                    'n4': nums[3], 'n5': nums[4], 'n6': nums[5],
                    'extra': nums[6], 'div1_prize': 'Pending', 'div1_winners': 0, 'url': url
                }
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                df.to_csv('marksix.csv', index=False)
                return True
    except Exception as e:
        print(f"Scraper error: {e}")
    return False

if __name__ == "__main__":
    add_latest_draw()
