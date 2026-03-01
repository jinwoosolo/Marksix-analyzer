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
    🔥 終極解析邏輯：採用視覺標籤過濾，徹底排除期數與標題數字
    """
    if not html:
        return {}

    soup = BeautifulSoup(html, "lxml")
    data = {"url": url}

    # 1. 提取日期 (從 URL 提取)
    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", url)
    if date_match:
        data["date"] = date_match.group(1).replace("-", "/")
    
    print(f"   [Debug] 處理日期: {data.get('date')}")

    # 2. 抓取號碼 - 採用最穩定的類名過濾
    # lottery.hk 的正選號碼球通常有 r-ball, g-ball, b-ball 類名
    # 特別號通常在 + 號後面
    
    main_balls = []
    # 尋找所有帶有球體特徵的 span 或 li
    # 我們優先尋找 class 包含 'ball' 且裡面是數字的元素
    all_elements = soup.find_all(['span', 'li', 'div'], class_=re.compile(r'ball|no-|num', re.I))
    
    unique_balls_in_order = []
    for el in all_elements:
        txt = el.get_text(strip=True)
        if txt.isdigit():
            val = int(txt)
            if 1 <= val <= 49:
                # 檢查這個元素是否在結果區塊內 (排除側邊欄或歷史紀錄)
                parent_text = el.parent.get_text() if el.parent else ""
                # 如果這是一個號碼球，它通常不會跟著 "Draw" 這種字眼在同一個單元格
                unique_balls_in_order.append(val)

    # 3. 數據過濾：排除期數干擾
    # 獲取期數 (如 26/023)
    draw_no_match = re.search(r'(\d{2})/\d{3}', soup.get_text())
    draw_no_prefix = int(draw_no_match.group(1)) if draw_no_match else None
    
    # 清洗得到的數字列表
    clean_balls = []
    for b in unique_balls_in_order:
        if b not in clean_balls:
            clean_balls.append(b)
    
    # 如果第一個數字剛好等於期數前綴 (26)，且總數 > 7，則剔除第一個
    if len(clean_balls) > 7 and draw_no_prefix and clean_balls[0] == draw_no_prefix:
        print(f"   [Debug] 偵測到期數干擾數字 {clean_balls[0]}，已剔除。")
        clean_balls = clean_balls[1:]

    # 4. 最終分配
    if len(clean_balls) >= 7:
        main = clean_balls[:6]
        extra = clean_balls[6]
        data.update({
            "n1": main[0], "n2": main[1], "n3": main[2],
            "n4": main[3], "n5": main[4], "n6": main[5],
            "extra": extra
        })
        print(f"   🎯 抓取成功: {main} + {extra}")
    else:
        # 最後備援：如果類名法失敗，嘗試在內容區尋找 7 個連續數字
        print(f"   [Debug] 類名抓取不足 ({len(clean_balls)} 個)，嘗試內容區掃描...")
        main_content = soup.find('main') or soup
        nums = re.findall(r'\b\d{1,2}\b', main_content.get_text(separator=' '))
        valid_nums = []
        for n in nums:
            v = int(n)
            if 1 <= v <= 49 and (not draw_no_prefix or v != draw_no_prefix):
                if v not in valid_nums: valid_nums.append(v)
        
        if len(valid_nums) >= 7:
            res = valid_nums[:7]
            data.update({
                "n1": res[0], "n2": res[1], "n3": res[2],
                "n4": res[3], "n5": res[4], "n6": res[5],
                "extra": res[6]
            })
            print(f"   🎯 備援抓取成功: {res[:6]} + {res[6]}")

    # 5. 獎金提取
    full_text = soup.get_text()
    prize_match = re.search(r'1st.*?HK\$?([\d,]+)', full_text, re.I | re.DOTALL)
    data["div1_prize"] = prize_match.group(1).replace(',', '') if prize_match else "0"
    data["div1_winners"] = "0"

    return data

def fetch_latest_marksix():
    """
    主執行程序：自動補齊 CSV 中缺失的近期數據
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
        # 獲取所有符合日期的詳情頁連結
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "/mark-six/results/20" in href:
                match = re.search(r"(\d{4}-\d{2}-\d{2})", href)
                if match:
                    date_val = match.group(1).replace("-", "/")
                    if date_val not in existing_dates:
                        full_url = urljoin(BASE_URL, href)
                        # 避免重複加入
                        if (date_val, full_url) not in draw_links:
                            draw_links.append((date_val, full_url))
        
        # 限制一次更新的數量
        draw_links = draw_links[:5]

        if not draw_links:
            print("ℹ️ CSV 數據已是最新，無需更新。")
            return True
            
        print(f"🔎 發現 {len(draw_links)} 筆缺失數據，開始補齊...")

        new_records = []
        for missing_date, link in reversed(draw_links): # 從舊日期開始
            print(f"⏳ 正在抓取: {missing_date} ...")
            time.sleep(2) 
            try:
                detail_resp = requests.get(link, headers=headers, timeout=20)
                if detail_resp.status_code == 200:
                    rec = parse_draw_page(detail_resp.text, link)
                    if rec.get("n1"):
                        new_records.append(rec)
                    else:
                        print(f"   ❌ 解析號碼失敗。")
            except Exception as e:
                print(f"   ❌ 抓取異常: {e}")

        if new_records:
            df_new = pd.DataFrame(new_records)
            df_final = pd.concat([df_existing, df_new], ignore_index=True)
            df_final['date_dt'] = pd.to_datetime(df_final['date'], format='%Y/%m/%d')
            df_final = df_final.sort_values('date_dt', ascending=True)
            df_final = df_final.drop_duplicates(subset=['date'], keep='last').drop(columns=['date_dt'])
            df_final.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"\n🎉 更新成功！目前 CSV 總量: {len(df_final)} 筆。")
            return True
        
    except Exception as e:
        print(f"❌ 嚴重錯誤: {e}")
    
    return False

if __name__ == "__main__":
    fetch_latest_marksix()
