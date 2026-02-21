import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import os
import time
from urllib.parse import urljoin
from datetime import datetime

# 基本配置
BASE_URL = "https://lottery.hk"
RESULTS_LIST_URL = "https://www.lottery.hk/en/mark-six/results"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def parse_draw_page(html, url):
    """精確解析詳情頁面，排除標題與日期數字的干擾"""
    if not html:
        return {}

    soup = BeautifulSoup(html, "lxml")
    data = {"url": url}

    # 1. 從 URL 提取日期 (最準確)
    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", url)
    if date_match:
        data["date"] = date_match.group(1).replace("-", "/")
    
    # 2. 定位主內容區，避開 Header 和 Nav (避免抓到標題中的期數和日期)
    main_content = soup.find('main') or soup.find('article') or soup.find('div', id='content')
    if not main_content:
        main_content = soup

    # 3. 抓取號碼球 - 針對 lottery.hk 的特定類名 (r-ball, g-ball, b-ball)
    balls = []
    
    # 搜尋帶有顏色特徵或 ball 字眼的標籤
    ball_elements = main_content.find_all(['span', 'li', 'div'], class_=re.compile(r'ball|r-b|g-b|b-b|no-', re.I))
    
    for el in ball_elements:
        txt = el.get_text(strip=True)
        # 確保是 1-49 的數字，且不包含雜訊文字
        if txt.isdigit():
            val = int(txt)
            if 1 <= val <= 49:
                balls.append(val)
    
    # 去重並保持出現順序
    unique_balls = []
    for b in balls:
        if b not in unique_balls:
            unique_balls.append(b)
            
    # 如果抓到太多（可能是包含下方歷史紀錄），我們只取最前面出現的一組 7 個
    if len(unique_balls) >= 7:
        res_balls = unique_balls[:7]
        main = res_balls[:6]
        extra = res_balls[6]
        data.update({
            "n1": main[0], "n2": main[1], "n3": main[2],
            "n4": main[3], "n5": main[4], "n6": main[5],
            "extra": extra
        })
        print(f"✅ 解析成功: {main} + {extra}")
    else:
        # 備選方案：如果類名抓不到，嘗試正則匹配區塊內的獨立數字
        # 我們找包含 "ball" 的容器
        print(f"⚠️ 類名抓取不足 ({len(unique_balls)} 個)，啟動 Regex 備援...")
        container = main_content.find('div', class_=re.compile(r'result|draw', re.I))
        if container:
            nums = re.findall(r'\b\d{1,2}\b', container.get_text(separator=' '))
            valid_nums = [int(n) for n in nums if 1 <= int(n) <= 49]
            if len(valid_nums) >= 7:
                unique_valid = []
                for n in valid_nums:
                    if n not in unique_valid: unique_valid.append(n)
                res_balls = unique_valid[:7]
                data.update({
                    "n1": res_balls[0], "n2": res_balls[1], "n3": res_balls[2],
                    "n4": res_balls[3], "n5": res_balls[4], "n6": res_balls[5],
                    "extra": res_balls[6]
                })

    # 獎金資訊
    full_text = soup.get_text()
    prize_match = re.search(r'1st.*?HK\$?([\d,]+).*?([\d.]+)', full_text, re.I | re.DOTALL)
    data["div1_prize"] = prize_match.group(1) if prize_match else "TBA"
    data["div1_winners"] = prize_match.group(2) if prize_match else 0

    return data

def update_marksix():
    print(f"🚀 啟動自動更新程序 (Source: lottery.hk)...")
    try:
        resp = requests.get(RESULTS_LIST_URL, headers=headers, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        
        # 1. 找到最新一期的詳情頁連結
        latest_link = None
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if re.search(r"/mark-six/results/20\d{2}-\d{2}-\d{2}", href):
                latest_link = urljoin(BASE_URL, href)
                break
        
        if not latest_link:
            print("❌ 找不到更新連結"); return

        print(f"🔍 進入最新期數頁面: {latest_link}")
        
        # 2. 檢查 CSV
        csv_path = 'marksix.csv'
        df = pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame()
        
        date_match = re.search(r"(\d{4}-\d{2}-\d{2})", latest_link)
        if date_match:
            target_date = date_match.group(1).replace("-", "/")
            if not df.empty and target_date in df['date'].astype(str).values:
                print(f"ℹ️ 日期 {target_date} 已在 CSV 中，略過。")
                return

        # 3. 抓取並解析
        time.sleep(2)
        detail_resp = requests.get(latest_link, headers=headers, timeout=20)
        rec = parse_draw_page(detail_resp.text, latest_link)

        if rec.get("n1") and rec.get("extra"):
            new_row = pd.DataFrame([rec])
            # 整合並排序
            df = pd.concat([df, new_row], ignore_index=True)
            df['dt'] = pd.to_datetime(df['date'], format='%Y/%m/%d')
            df = df.sort_values('dt', ascending=True).drop(columns=['dt'])
            df.to_csv(csv_path, index=False)
            print(f"🎉 數據更新成功: {rec['date']}")
        else:
            print("❌ 號碼解析失敗，請檢查網頁結構。")
            
    except Exception as e:
        print(f"❌ 發生錯誤: {e}")

if __name__ == "__main__":
    update_marksix()
