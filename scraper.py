import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import os
import time
from urllib.parse import urljoin
from datetime import datetime

# 配置與 Header
BASE_URL = "https://lottery.hk"
RESULTS_URL = "https://www.lottery.hk/en/mark-six/results"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def parse_draw_page(html, url):
    """使用你提供的高級解析邏輯，確保抓到號碼"""
    if not html:
        return {}

    soup = BeautifulSoup(html, "lxml")
    data = {"url": url}

    # 從 URL 提取日期
    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", url)
    if date_match:
        data["date"] = date_match.group(1).replace("-", "/")
    else:
        # 如果 URL 沒日期，嘗試從頁面內容找
        content = soup.get_text()
        d_match = re.search(r'(\d{1,2})\s+([A-Z][a-z]+)\s+(\d{4})', content)
        if d_match:
            # 簡單處理日期 (scraper 不需要處理 2002 年歷史，只處理最新)
            try:
                dt = datetime.strptime(d_match.group(0), '%d %B %Y')
                data["date"] = dt.strftime('%Y/%m/%d')
            except: pass

    # --- 策略 1：掃描所有數字元素 ---
    all_numbers = []
    selectors = ["li", "span", "div[class*='ball']", "div[class*='number']", ".result-numbers"]
    for selector in selectors:
        for elem in soup.select(selector):
            txt = elem.get_text(strip=True)
            if txt.isdigit():
                n = int(txt)
                if 1 <= n <= 49:
                    all_numbers.append(n)

    # 去重
    seen = set()
    unique_ordered = []
    for n in all_numbers:
        if n not in seen:
            seen.add(n)
            unique_ordered.append(n)

    if len(unique_ordered) >= 7:
        main = unique_ordered[:6]
        extra = unique_ordered[6]
        data.update({
            "n1": main[0], "n2": main[1], "n3": main[2],
            "n4": main[3], "n5": main[4], "n6": main[5],
            "extra": extra
        })
        print(f"✅ 解析成功: {main} + {extra}")

    # --- 策略 2：Regex Fallback ---
    if "n1" not in data:
        full_text = re.sub(r'[^\w\s\d/+]', ' ', soup.get_text())
        patterns = [
            r'(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s*[\+\s]+(\d{1,2})',
            r'(\d+)\D+(\d+)\D+(\d+)\D+(\d+)\D+(\d+)\D+(\d+)\D*\+\D*(\d+)'
        ]
        for pattern in patterns:
            match = re.search(pattern, full_text, re.I)
            if match:
                nums = [int(match.group(i)) for i in range(1, 8)]
                data.update({
                    "n1": nums[0], "n2": nums[1], "n3": nums[2],
                    "n4": nums[3], "n5": nums[4], "n6": nums[5],
                    "extra": nums[6]
                })
                break

    # 獎金資訊 (選填)
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
    print(f"🚀 開始檢查更新: {datetime.now()}")
    
    # 1. 抓取主列表頁面
    try:
        resp = requests.get(RESULTS_URL, headers=headers, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
    except Exception as e:
        print(f"❌ 無法連接首頁: {e}"); return

    # 2. 找到最上面的第一期連結
    latest_link = None
    for a in soup.find_all("a", href=True):
        if "/mark-six/results/20" in a["href"]: # 找包含年份的結果連結
            latest_link = urljoin(BASE_URL, a["href"])
            break

    if not latest_link:
        print("❌ 找不到最新的開彩連結"); return

    print(f"🔍 發現最新開彩頁面: {latest_link}")

    # 3. 檢查日期是否已經在 CSV
    csv_path = 'marksix.csv'
    df = pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame()
    
    date_in_url = re.search(r"(\d{4}-\d{2}-\d{2})", latest_link)
    if date_in_url:
        check_date = date_in_url.group(1).replace("-", "/")
        if not df.empty and check_date in df['date'].astype(str).values:
            print(f"ℹ️ {check_date} 數據已存在，無需操作。")
            return

    # 4. 抓取詳情頁並解析
    try:
        detail_resp = requests.get(latest_link, headers=headers, timeout=20)
        rec = parse_draw_page(detail_resp.text, latest_link)
    except Exception as e:
        print(f"❌ 抓取詳情頁失敗: {e}"); return

    if rec.get("date") and rec.get("n1"):
        # 再次確認日期
        if not df.empty and rec["date"] in df['date'].astype(str).values:
            print(f"ℹ️ {rec['date']} 數據已在 CSV 中。")
            return

        # 寫入 CSV
        new_row = pd.DataFrame([rec])
        df = pd.concat([df, new_row], ignore_index=True)
        # 排序
        df['dt'] = pd.to_datetime(df['date'], format='%Y/%m/%d')
        df = df.sort_values('dt', ascending=True).drop(columns=['dt'])
        df.to_csv(csv_path, index=False)
        print(f"✅ 成功更新 CSV！日期: {rec['date']}")
    else:
        print("❌ 解析失敗，未能獲取完整數據。")

if __name__ == "__main__":
    update_marksix()
