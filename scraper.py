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
    """精確解析詳情頁面，排除標題干擾"""
    if not html:
        return {}

    soup = BeautifulSoup(html, "lxml")
    data = {"url": url}

    # 1. 從 URL 提取日期 (格式: 2026-02-21)
    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", url)
    if not date_match:
        return {}
    
    raw_date = date_match.group(1)
    formatted_date = raw_date.replace("-", "/")
    data["date"] = formatted_date
    
    # 提取日期的部分數字，用來做過濾 (例如 21, 02, 2026)
    date_parts = set(re.findall(r"\d+", raw_date))

    # 2. 策略：尋找真正的號碼球容器
    # 在 lottery.hk 詳情頁，號碼通常在一個 class 包含 'balls' 或 'result-box' 的容器內
    balls = []
    
    # 優先尋找結果列表標籤
    ball_container = soup.find(['ul', 'div'], class_=re.compile(r'ball|result-numbers|results-list', re.I))
    
    if ball_container:
        # 尋找容器內的數字
        for el in ball_container.find_all(['li', 'span', 'div']):
            txt = el.get_text(strip=True)
            cls = "".join(el.get('class', []))
            # 號碼球通常內容是數字，且 class 包含 ball 或 no-
            if txt.isdigit() and re.search(r'ball|no-|num', cls, re.I):
                val = int(txt)
                if 1 <= val <= 49:
                    balls.append(val)

    # 3. 如果沒抓到，使用備選方案（掃描全頁但過濾干擾）
    if len(balls) < 7:
        temp_balls = []
        # 尋找所有可能是球的元素
        for el in soup.find_all(['span', 'li', 'div'], class_=re.compile(r'ball|no-|num', re.I)):
            txt = el.get_text(strip=True)
            if txt.isdigit():
                val = int(txt)
                # 過濾邏輯：排除掉出現在 URL/標題中的日期數字和期數數字
                # 期數通常也會出現在文本中，這裡假設號碼球在頁面中後段
                if 1 <= val <= 49:
                    temp_balls.append(val)
        
        # 排除重複並保持順序
        unique_balls = []
        for b in temp_balls:
            if b not in unique_balls:
                unique_balls.append(b)
        
        # 如果數字過多，通常真正的號碼球會排在一起
        # 我們尋找連續出現的 7 個號碼
        if len(unique_balls) >= 7:
            # 觀察發現標題數字通常在最前面，我們從後面開始取可能是正確的，
            # 但更好的方法是找特定的父容器
            balls = unique_balls[-7:]

    if len(balls) >= 7:
        # 為了確保萬一抓到前面的期數，我們做最後檢查
        # 如果前幾個數字剛好是年份或日期，則往後移
        if len(balls) > 7:
            # 排除掉跟日期太像的數字
            balls = [b for b in balls if str(b) not in date_parts]
        
        if len(balls) >= 7:
            # 取最後 7 個通常是最準確的號碼球
            res_balls = balls[-7:]
            main = res_balls[:6]
            extra = res_balls[6]
            data.update({
                "n1": main[0], "n2": main[1], "n3": main[2],
                "n4": main[3], "n5": main[4], "n6": main[5],
                "extra": extra
            })
            print(f"✅ 成功提取號碼: {main} + {extra}")
        else:
            print(f"❌ 號碼過濾後不足: {balls}")

    # 獎金資訊
    full_text = soup.get_text()
    prize_match = re.search(r'1st.*?HK\$?([\d,]+).*?([\d.]+)', full_text, re.I | re.DOTALL)
    data["div1_prize"] = prize_match.group(1) if prize_match else "TBA"
    data["div1_winners"] = prize_match.group(2) if prize_match else 0

    return data

def update_marksix():
    print(f"🚀 啟動自動更新程序...")
    try:
        resp = requests.get(RESULTS_LIST_URL, headers=headers, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        
        # 找到最新一期的連結
        latest_link = None
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if re.search(r"/mark-six/results/20\d{2}-\d{2}-\d{2}", href):
                latest_link = urljoin(BASE_URL, href)
                break
        
        if not latest_link:
            print("❌ 找不到更新連結"); return

        print(f"🔍 進入最新期數頁面: {latest_link}")
        
        # 檢查是否已存在
        csv_path = 'marksix.csv'
        df = pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame()
        
        date_match = re.search(r"(\d{4}-\d{2}-\d{2})", latest_link)
        if date_match:
            target_date = date_match.group(1).replace("-", "/")
            if not df.empty and target_date in df['date'].astype(str).values:
                print(f"ℹ️ 日期 {target_date} 已在 CSV 中，略過更新。")
                return

        # 抓取並解析
        time.sleep(1)
        detail_resp = requests.get(latest_link, headers=headers, timeout=20)
        rec = parse_draw_page(detail_resp.text, latest_link)

        if rec.get("n1"):
            new_row = pd.DataFrame([rec])
            df = pd.concat([df, new_row], ignore_index=True)
            df['dt'] = pd.to_datetime(df['date'], format='%Y/%m/%d')
            df = df.sort_values('dt', ascending=True).drop(columns=['dt'])
            df.to_csv(csv_path, index=False)
            print(f"🎉 數據更新成功: {rec['date']}")
        else:
            print("❌ 號碼解析失敗。")
            
    except Exception as e:
        print(f"❌ 發生錯誤: {e}")

if __name__ == "__main__":
    update_marksix()
