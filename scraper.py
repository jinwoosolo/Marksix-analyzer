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
RESULTS_URL = "https://www.lottery.hk/en/mark-six/results"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def parse_draw_page(html, url):
    """採用你提供的高級解析邏輯，確保精確抓取 n1-n6 + extra"""
    if not html:
        return {}

    soup = BeautifulSoup(html, "lxml")
    data = {"url": url}

    # 1. 從 URL 提取日期 (格式: 2026-02-21)
    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", url)
    if date_match:
        data["date"] = date_match.group(1).replace("-", "/")
    
    # 2. 策略：掃描所有數字元素 (你提供的核心邏輯)
    all_numbers = []
    # 增加了選擇器精確度
    selectors = ["li", "span", "div[class*='ball']", "div[class*='number']", ".result-numbers"]
    for selector in selectors:
        for elem in soup.select(selector):
            txt = elem.get_text(strip=True)
            if txt.isdigit():
                n = int(txt)
                if 1 <= n <= 49:
                    all_numbers.append(n)

    # 去重並保持順序
    seen = set()
    unique_ordered = []
    for n in all_numbers:
        if n not in seen:
            seen.add(n)
            unique_ordered.append(n)

    # 如果有 7 個或以上號碼，取前 6 個為正獎，第 7 個為特別獎
    if len(unique_ordered) >= 7:
        main = unique_ordered[:6]
        extra = unique_ordered[6]
        data.update({
            "n1": main[0], "n2": main[1], "n3": main[2],
            "n4": main[3], "n5": main[4], "n6": main[5],
            "extra": extra
        })
        print(f"✅ 解析成功: {main} + {extra}")
        
    # 如果解析不到日期，從頁面文本備份尋找
    if "date" not in data:
        full_text = soup.get_text()
        d_match = re.search(r'(\d{1,2})\s+([A-Z][a-z]+)\s+(\d{4})', full_text)
        if d_match:
            try:
                dt = datetime.strptime(d_match.group(0), '%d %B %Y')
                data["date"] = dt.strftime('%Y/%m/%d')
            except: pass

    # 獎金資訊
    full_text = soup.get_text()
    prize_match = re.search(r'1st.*?HK\$?([\d,]+).*?([\d.]+)', full_text, re.I | re.DOTALL)
    if prize_match:
        data["div1_prize"] = prize_match.group(1)
        data["div1_winners"] = prize_match.group(2)
    else:
        data["div1_prize"] = "TBA"
        data["div1_winners"] = 0

    return data

def update_marksix():
    print(f"🚀 開始執行每日更新...")
    
    try:
        resp = requests.get(RESULTS_URL, headers=headers, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
    except Exception as e:
        print(f"❌ 無法連接首頁: {e}"); return

    # 找到最新一期的連結 (通常是結果頁的第一個結果連結)
    latest_link = None
    for a in soup.find_all("a", href=True):
        if "/mark-six/results/20" in a["href"]:
            latest_link = urljoin(BASE_URL, a["href"])
            break

    if not latest_link:
        print("❌ 找不到最新的結果連結"); return

    print(f"🔍 最新一期頁面: {latest_link}")

    csv_path = 'marksix.csv'
    df = pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame()
    
    # 檢查是否已存在
    date_in_url = re.search(r"(\d{4}-\d{2}-\d{2})", latest_link)
    if date_in_url:
        check_date = date_in_url.group(1).replace("-", "/")
        if not df.empty and check_date in df['date'].astype(str).values:
            print(f"ℹ️ 數據 {check_date} 已在 CSV 中，無需更新。")
            return

    # 抓取詳情頁
    try:
        detail_resp = requests.get(latest_link, headers=headers, timeout=20)
        rec = parse_draw_page(detail_resp.text, latest_link)
    except Exception as e:
        print(f"❌ 詳情頁抓取失敗: {e}"); return

    if rec.get("date") and rec.get("n1"):
        if not df.empty and rec["date"] in df['date'].astype(str).values:
            print(f"ℹ️ 數據 {rec['date']} 已存在。")
            return

        # 寫入
        new_row = pd.DataFrame([rec])
        df = pd.concat([df, new_row], ignore_index=True)
        # 排序
        df['dt'] = pd.to_datetime(df['date'], format='%Y/%m/%d')
        df = df.sort_values('dt', ascending=True).drop(columns=['dt'])
        df.to_csv(csv_path, index=False)
        print(f"✅ CSV 更新完成: {rec['date']}")
    else:
        print("❌ 解析失敗，找不到有效號碼。")

if __name__ == "__main__":
    update_marksix()
