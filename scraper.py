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
    🔥 終極精確解析：徹底排除日期、年份及期數干擾
    """
    if not html:
        return {}

    soup = BeautifulSoup(html, "lxml")
    data = {"url": url}

    # 1. 提取日期並建立「黑名單數字」
    date_match = re.search(r"(\d{4})-(\d{2})-(\d{2})", url)
    if date_match:
        year_full, month_raw, day_raw = date_match.groups()
        data["date"] = f"{year_full}/{month_raw}/{day_raw}"
        
        # 黑名單：包含年份(2026)、日期(28)、月份(02)
        # 同時包含年份後兩位(26)，因為這經常出現在期數編號中
        blacklist = {
            year_full, 
            str(int(year_full[2:])), 
            str(int(month_raw)), 
            str(int(day_raw)),
            "2026", "26", "28", "21" # 強制加入已知干擾項
        }
    else:
        return {}

    print(f"   [Debug] 處理日期: {data.get('date')} | 排除清單: {blacklist}")

    # 2. 鎖定結果表格區域
    # 我們尋找包含號碼球的最核心容器
    balls = []
    
    # 策略 A: 優先抓取具有 'ball' 樣式的元素，但必須在結果容器內
    # 這樣可以避開導航欄或頁尾的數字
    content_area = soup.select_one(".result-numbers") or \
                   soup.select_one(".balls") or \
                   soup.select_one(".draw-results") or \
                   soup.select_one(".result-box") or \
                   soup.select_one("main")

    if content_area:
        # 尋找帶有 'ball' 類名的元素
        ball_elements = content_area.find_all(['span', 'li', 'div'], class_=re.compile(r'ball|no-', re.I))
        
        candidates = []
        for el in ball_elements:
            txt = el.get_text(strip=True)
            if txt.isdigit():
                val_str = str(int(txt))
                # 初步過濾：1-49 且不在黑名單中（除非黑名單數字本身就是中獎號碼）
                # 注意：如果數字帶有明顯的球體類名 (如 r-ball)，即使在黑名單也應保留
                cls = "".join(el.get('class', []))
                is_real_ball = re.search(r'r-b|g-b|b-b|ball', cls, re.I)
                
                if 1 <= int(val_str) <= 49:
                    if is_real_ball or val_str not in blacklist:
                        candidates.append(int(val_str))

        # 數據清洗：去重
        for c in candidates:
            if c not in balls:
                balls.append(c)

    # 3. 備援與二度檢查 (針對 28 號多出一個 28 的問題)
    if len(balls) > 7:
        # 如果抓到 8 個數字且第一個跟日期一樣，則它是誤抓
        if str(balls[0]) == str(int(day_raw)):
            print(f"   [Debug] 偵測到首位數字與日期重複 ({balls[0]})，執行修正位移。")
            balls = balls[1:]

    # 4. 數據分配 (取前 7 個有效數字)
    if len(balls) >= 7:
        res = balls[:7]
        data.update({
            "n1": res[0], "n2": res[1], "n3": res[2],
            "n4": res[3], "n5": res[4], "n6": res[5],
            "extra": res[6]
        })
        print(f"   🎯 解析成功: {res[:6]} + {res[6]}")
    else:
        print(f"   ❌ 解析失敗，僅抓到: {balls}")

    # 5. 獎金
    full_text = soup.get_text()
    prize_match = re.search(r'1st.*?HK\$?([\d,]+)', full_text, re.I | re.DOTALL)
    data["div1_prize"] = prize_match.group(1).replace(',', '') if prize_match else "0"
    data["div1_winners"] = "0"

    return data

def fetch_latest_marksix():
    """
    主執行程序：自動補齊 CSV 缺失數據
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
        # 獲取近期所有連結
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
        
        # 每次最多補 5 筆
        draw_links = draw_links[:5]

        if not draw_links:
            print("ℹ️ 所有數據已是最新。")
            return True
            
        print(f"🔎 發現 {len(draw_links)} 筆需要補齊的數據...")

        new_records = []
        for missing_date, link in reversed(draw_links):
            print(f"⏳ 正在處理: {missing_date} ...")
            time.sleep(2) 
            try:
                detail_resp = requests.get(link, headers=headers, timeout=20)
                if detail_resp.status_code == 200:
                    rec = parse_draw_page(detail_resp.text, link)
                    if rec.get("n1"):
                        new_records.append(rec)
                    else:
                        print(f"   ❌ {missing_date} 解析失敗。")
            except Exception as e:
                print(f"   ❌ 抓取錯誤: {e}")

        if new_records:
            df_new = pd.DataFrame(new_records)
            df_final = pd.concat([df_existing, df_new], ignore_index=True)
            df_final['date_dt'] = pd.to_datetime(df_final['date'], format='%Y/%m/%d')
            df_final = df_final.sort_values('date_dt', ascending=True)
            df_final = df_final.drop_duplicates(subset=['date'], keep='last').drop(columns=['date_dt'])
            df_final.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"\n🎉 數據同步完成！")
            return True
        
    except Exception as e:
        print(f"❌ 嚴重錯誤: {e}")
    
    return False

if __name__ == "__main__":
    fetch_latest_marksix()
