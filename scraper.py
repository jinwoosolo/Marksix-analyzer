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
    🔥 終極解析邏輯：利用 CSS 類名精確定位號碼球，徹底排除日期與期數文字
    """
    if not html:
        return {}

    soup = BeautifulSoup(html, "lxml")
    data = {"url": url}

    # 1. 提取日期 (從 URL 提取)
    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", url)
    if date_match:
        raw_date = date_match.group(1)
        data["date"] = raw_date.replace("-", "/")
        # 提取日期中的數字 (2026, 02, 28 等)，用於稍後排除
        date_parts = [str(int(x)) for x in re.findall(r'\d+', raw_date)]
    else:
        return {}

    print(f"   [Debug] 處理日期: {data.get('date')}")

    # 2. 定位結果核心區塊
    # 我們尋找 class 包含 'balls' 或 'draw-results' 的容器
    ball_area = soup.select_one(".result-numbers") or \
                soup.select_one(".balls") or \
                soup.select_one(".draw-results") or \
                soup.select_one(".result-box")

    balls = []
    if ball_area:
        # 關鍵：只抓取具有 'ball' 類名的元素 (r-ball, g-ball, b-ball)
        # 這些類名專門用於顯示號碼球，標題和期數文字不會使用這些類名
        elements = ball_area.find_all(['span', 'li', 'div'], class_=re.compile(r'ball|no-', re.I))
        for el in elements:
            txt = el.get_text(strip=True)
            if txt.isdigit():
                val = int(txt)
                if 1 <= val <= 49 and val not in balls:
                    balls.append(val)
                    
    # 3. 備援邏輯：如果標籤法失效，使用極其嚴格的過濾規則
    if len(balls) < 7:
        print(f"   [Debug] 標籤法不足 ({len(balls)} 個)，啟動嚴格過援...")
        main_content = soup.find('main') or soup
        # 獲取期數前綴 (如 26/023 中的 26)
        draw_no_match = re.search(r'(\d{2})/\d{3}', soup.get_text())
        draw_no = draw_no_match.group(1) if draw_no_match else None
        
        all_nums = re.findall(r'\b\d{1,2}\b', main_content.get_text(separator=' '))
        temp_list = []
        for n in all_nums:
            v = int(n)
            # 過濾掉：日期數字、期數數字
            if 1 <= v <= 49 and n not in date_parts and n != draw_no:
                if v not in temp_list:
                    temp_list.append(v)
        
        if len(temp_list) >= 7:
            balls = temp_list[:7]

    # 4. 分配數據
    if len(balls) >= 7:
        # 取前 7 個符合條件的數字
        res = balls[:7]
        data.update({
            "n1": res[0], "n2": res[1], "n3": res[2],
            "n4": res[3], "n5": res[4], "n6": res[5],
            "extra": res[6]
        })
        print(f"   🎯 成功提取: {res[:6]} + {res[6]}")
    else:
        print(f"   ❌ 解析失敗，抓到的數字: {balls}")

    # 5. 獎金提取
    full_text = soup.get_text()
    prize_match = re.search(r'1st.*?HK\$?([\d,]+)', full_text, re.I | re.DOTALL)
    data["div1_prize"] = prize_match.group(1).replace(',', '') if prize_match else "0"
    data["div1_winners"] = "0"

    return data

def fetch_latest_marksix():
    """
    主執行程序：自動補齊 CSV 中缺失或錯誤的數據
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

        resp = requests.get(RESULTS_LIST_URL, headers=headers, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        
        draw_links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "/mark-six/results/20" in href:
                match = re.search(r"(\d{4}-\d{2}-\d{2})", href)
                if match:
                    date_val = match.group(1).replace("-", "/")
                    if date_val not in existing_dates:
                        full_url = urljoin(BASE_URL, href)
                        if (date_val, full_url) not in draw_links:
                            draw_links.append((date_val, full_url))
        
        draw_links = draw_links[:5]

        if not draw_links:
            print("ℹ️ CSV 數據已是最新。")
            return True
            
        print(f"🔎 發現 {len(draw_links)} 筆缺失數據，開始補齊...")

        new_records = []
        for missing_date, link in reversed(draw_links):
            print(f"⏳ 正在抓取: {missing_date} ...")
            time.sleep(2) 
            try:
                detail_resp = requests.get(link, headers=headers, timeout=20)
                if detail_resp.status_code == 200:
                    rec = parse_draw_page(detail_resp.text, link)
                    if rec.get("n1"):
                        new_records.append(rec)
                    else:
                        print(f"   ❌ 解析失敗。")
            except Exception as e:
                print(f"   ❌ 抓取異常: {e}")

        if new_records:
            df_new = pd.DataFrame(new_records)
            df_final = pd.concat([df_existing, df_new], ignore_index=True)
            df_final['date_dt'] = pd.to_datetime(df_final['date'], format='%Y/%m/%d')
            df_final = df_final.sort_values('date_dt', ascending=True)
            df_final = df_final.drop_duplicates(subset=['date'], keep='last').drop(columns=['date_dt'])
            df_final.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"\n🎉 更新成功！")
            return True
        
    except Exception as e:
        print(f"❌ 嚴重錯誤: {e}")
    
    return False

if __name__ == "__main__":
    fetch_latest_marksix()
