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

    # 1. 從 URL 提取日期 (格式: 2026-02-21)
    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", url)
    if not date_match:
        return {}
    
    raw_date = date_match.group(1)
    formatted_date = raw_date.replace("-", "/")
    data["date"] = formatted_date
    
    # 2. 策略：尋找真正的號碼球
    # lottery.hk 詳情頁中，真正的號碼通常在一個特定的 <ul> 內
    # 我們尋找 class 包含 'balls' 或 'mark-six-result' 的區域
    balls = []
    
    # 搜尋特定的號碼列表
    # 優先尋找 class 包含 'balls' 的 <ul> 或 <div>
    target_area = soup.find(['ul', 'div'], class_=re.compile(r'balls|result-box|draw-results', re.I))
    
    if not target_area:
        # 如果找不到特定區域，搜尋第一個具有多個球的容器
        for container in soup.find_all(['ul', 'div']):
            ball_elements = container.find_all(['li', 'span'], class_=re.compile(r'ball|no-', re.I))
            if len(ball_elements) >= 7:
                target_area = container
                break

    if target_area:
        # 在目標區域內抓取號碼
        for el in target_area.find_all(['li', 'span', 'div']):
            txt = el.get_text(strip=True)
            cls = "".join(el.get('class', []))
            
            # 號碼球特徵：是純數字，且 class 包含 'ball' 或 'no-' (代表號碼球圖示)
            if txt.isdigit() and re.search(r'ball|no-|num', cls, re.I):
                val = int(txt)
                if 1 <= val <= 49:
                    balls.append(val)
        
        # 去重（有些結構會重複渲染）
        unique_balls = []
        for b in balls:
            if b not in unique_balls:
                unique_balls.append(b)
        balls = unique_balls

    # 3. 備選方案：如果標籤識別不到，但頁面有 7 個以上的球狀物
    if len(balls) < 7:
        # 直接搜尋全頁所有帶球樣式的號碼
        fallback_balls = []
        for el in soup.find_all(['li', 'span'], class_=re.compile(r'ball|no-', re.I)):
            txt = el.get_text(strip=True)
            if txt.isdigit():
                val = int(txt)
                if 1 <= val <= 49:
                    fallback_balls.append(val)
        
        # 排除掉頁面上可能重複的歷史紀錄（通常前 7 個是本期結果）
        if len(fallback_balls) >= 7:
            # 詳情頁中最顯眼的號碼通常就是結果
            balls = fallback_balls[:7]

    if len(balls) >= 7:
        # 確保我們拿到的是本期的 6+1
        main = balls[:6]
        extra = balls[6]
        data.update({
            "n1": main[0], "n2": main[1], "n3": main[2],
            "n4": main[3], "n5": main[4], "n6": main[5],
            "extra": extra
        })
        print(f"✅ 成功提取號碼: {main} + {extra}")
    else:
        # 如果最後還是失敗，印出當前抓到的東西輔助除錯
        print(f"⚠️ 解析警告: 偵測到數字不足 ({len(balls)} 個): {balls}")

    # 獎金資訊
    full_text = soup.get_text()
    prize_match = re.search(r'1st.*?HK\$?([\d,]+).*?([\d.]+)', full_text, re.I | re.DOTALL)
    data["div1_prize"] = prize_match.group(1) if prize_match else "TBA"
    data["div1_winners"] = prize_match.group(2) if prize_match else 0

    return data

def update_marksix():
    print(f"🚀 啟動自動更新程序 (Target: lottery.hk)...")
    try:
        resp = requests.get(RESULTS_LIST_URL, headers=headers, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        
        # 1. 找到最新一期的連結 (第一個符合日期的連結)
        latest_link = None
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if re.search(r"/mark-six/results/20\d{2}-\d{2}-\d{2}", href):
                latest_link = urljoin(BASE_URL, href)
                break
        
        if not latest_link:
            print("❌ 找不到更新連結"); return

        print(f"🔍 進入最新期數頁面: {latest_link}")
        
        # 2. 檢查 CSV 是否已存在該日期
        csv_path = 'marksix.csv'
        df = pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame()
        
        date_match = re.search(r"(\d{4}-\d{2}-\d{2})", latest_link)
        if date_match:
            target_date = date_match.group(1).replace("-", "/")
            if not df.empty and target_date in df['date'].astype(str).values:
                print(f"ℹ️ 日期 {target_date} 已在 CSV 中，略過更新。")
                return

        # 3. 抓取並解析
        time.sleep(1.5) # 增加延遲，確保頁面加載
        detail_resp = requests.get(latest_link, headers=headers, timeout=20)
        rec = parse_draw_page(detail_resp.text, latest_link)

        if rec.get("n1") and rec.get("extra"):
            new_row = pd.DataFrame([rec])
            df = pd.concat([df, new_row], ignore_index=True)
            # 確保日期格式一致並排序
            df['dt'] = pd.to_datetime(df['date'], format='%Y/%m/%d')
            df = df.sort_values('dt', ascending=True).drop(columns=['dt'])
            df.to_csv(csv_path, index=False)
            print(f"🎉 數據更新成功: {rec['date']} | {rec['n1']},{rec['n2']},{rec['n3']},{rec['n4']},{rec['n5']},{rec['n6']} + {rec['extra']}")
        else:
            print("❌ 號碼解析失敗，未找到完整的 6+1 號碼。")
            
    except Exception as e:
        print(f"❌ 發生錯誤: {e}")

if __name__ == "__main__":
    update_marksix()
