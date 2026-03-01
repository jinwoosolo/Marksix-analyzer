import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import os
import time
from urllib.parse import urljoin
from datetime import datetime

# --- 全域配置 ---
BASE_URL = "https://www.lottery.hk"
RESULTS_LIST_URL = "https://www.lottery.hk/en/mark-six/results"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
}

def parse_draw_page(html, url):
    """
    🔥 修正版解析邏輯：嚴格隔離「期數數字」與「號碼球數字」
    """
    if not html:
        return {}

    soup = BeautifulSoup(html, "lxml")
    data = {"url": url}

    # 1. 提取日期 (從 URL 提取最準確)
    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", url)
    if date_match:
        data["date"] = date_match.group(1).replace("-", "/")
    
    # 2. 定位結果核心區塊
    # 我們尋找 class 包含 'balls' 或 'draw-results' 的容器，這能有效避開左側的 Draw Number
    balls = []
    
    # 優先鎖定專門存放號碼球的 <ul> 或 <div>
    target_area = soup.select_one(".result-numbers") or \
                  soup.select_one(".balls") or \
                  soup.select_one(".results-list") or \
                  soup.select_one(".draw-results")
    
    if not target_area:
        # 如果找不到特定容器，則搜尋 main 區域，但避開表格的第一欄 (通常是期數)
        target_area = soup.select_one("main") or soup

    # 3. 抓取號碼 - 嚴格限制：必須是具有「球體樣式」的元素
    # 我們搜尋 class 包含 'ball' 的標籤，這樣就不會抓到普通文本裡的 '26/023'
    ball_elements = target_area.find_all(['li', 'span', 'div'], class_=re.compile(r'ball|no-|num', re.I))
    
    temp_balls = []
    for el in ball_elements:
        txt = el.get_text(strip=True)
        if txt.isdigit():
            val = int(txt)
            if 1 <= val <= 49:
                temp_balls.append(val)
    
    # 4. 數據檢查與分開正獎/特別獎
    # lottery.hk 詳情頁通常只有一組 7 個球。如果抓到多組，通常第一組是正確的。
    if len(temp_balls) >= 7:
        # 再次檢查：如果第一個數字跟期數的前兩位(26)一樣，且位置可疑，我們檢查是否有多餘數字
        # 但在詳情頁中，只要我們鎖定了 .balls 容器，就不應該抓到期數。
        balls = temp_balls[:7]
        main = balls[:6]
        extra = balls[6]
        data.update({
            "n1": main[0], "n2": main[1], "n3": main[2],
            "n4": main[3], "n5": main[4], "n6": main[5],
            "extra": extra
        })
    
    # 5. 獎金提取
    full_text = soup.get_text()
    prize_match = re.search(r'1st.*?HK\$?([\d,]+)', full_text, re.I | re.DOTALL)
    data["div1_prize"] = prize_match.group(1).replace(',', '') if prize_match else "0"
    data["div1_winners"] = "0"

    return data

def fetch_latest_marksix():
    """
    主執行程序：自動補齊或修正 CSV 中缺失/錯誤的數據
    """
    print(f"\n{'='*60}")
    print(f"🚀 六合彩精確抓取器啟動: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    try:
        csv_path = 'marksix.csv'
        if os.path.exists(csv_path):
            df_existing = pd.read_csv(csv_path)
            df_existing['date'] = df_existing['date'].astype(str)
            existing_dates = set(df_existing['date'].values)
        else:
            df_existing = pd.DataFrame(columns=['date', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'extra', 'div1_prize', 'div1_winners', 'url'])
            existing_dates = set()

        # 1. 存取列表頁
        resp = requests.get(RESULTS_LIST_URL, headers=headers, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        
        # 2. 搜尋所有開獎連結
        draw_links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "/mark-six/results/20" in href:
                match = re.search(r"(\d{4}-\d{2}-\d{2})", href)
                if match:
                    date_val = match.group(1).replace("-", "/")
                    # 如果日期不在 CSV，或者您想強制更新最近幾期，可以在這裡調整
                    if date_val not in existing_dates:
                        full_url = urljoin(BASE_URL, href)
                        draw_links.append((date_val, full_url))

        if not draw_links:
            print("ℹ️ CSV 數據已是最新。")
            return True
            
        print(f"🔎 發現 {len(draw_links)} 筆缺失或待修正數據。")

        new_records = []
        for missing_date, link in reversed(draw_links): # 從舊到新補
            print(f"⏳ 正在精確抓取: {missing_date} ...")
            time.sleep(2) 
            try:
                detail_resp = requests.get(link, headers=headers, timeout=20)
                if detail_resp.status_code == 200:
                    rec = parse_draw_page(detail_resp.text, link)
                    if rec.get("n1"):
                        new_records.append(rec)
                        print(f"   ✅ 成功抓取: {rec['n1']}, {rec['n2']}, {rec['n3']}, {rec['n4']}, {rec['n5']}, {rec['n6']} + {rec['extra']}")
                    else:
                        print(f"   ⚠️ 解析失敗。")
            except Exception as e:
                print(f"   ❌ 抓取錯誤: {e}")

        # 3. 合併並儲存
        if new_records:
            df_new = pd.DataFrame(new_records)
            df_final = pd.concat([df_existing, df_new], ignore_index=True)
            df_final['date_dt'] = pd.to_datetime(df_final['date'], format='%Y/%m/%d')
            df_final = df_final.sort_values('date_dt', ascending=True)
            df_final = df_final.drop_duplicates(subset=['date'], keep='last').drop(columns=['date_dt'])
            df_final.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"\n🎉 更新完成！目前 CSV 總量: {len(df_final)} 筆。")
            return True
        
    except Exception as e:
        print(f"❌ 嚴重錯誤: {e}")
    
    return False

if __name__ == "__main__":
    fetch_latest_marksix()
