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
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
}

def parse_draw_page(html, url):
    """精確解析詳情頁面，排除標題、日期、期數的數字干擾"""
    if not html:
        return {}

    soup = BeautifulSoup(html, "lxml")
    data = {"url": url}

    # 1. 從 URL 提取日期 (最準確的基礎)
    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", url)
    if not date_match:
        return {}
    
    raw_date = date_match.group(1)
    formatted_date = raw_date.replace("-", "/")
    data["date"] = formatted_date
    
    # 提取日期組成部分 (如 21, 02, 2026)，稍後用來過濾誤抓
    date_parts = set(re.findall(r"\d+", raw_date))
    # 增加年份後兩位 (26) 到過濾名單
    date_parts.add(raw_date[2:4])

    # 2. 尋找主結果區塊 (避開側邊欄和頁尾)
    main_section = soup.find('main') or soup.find('article') or soup.find('div', id='content')
    if not main_section:
        main_section = soup

    # 3. 策略：鎖定存放號碼的 <ul> 或 <div> 容器
    # lottery.hk 常用 class: results-list, balls, mark-six-result
    balls = []
    
    # 我們找尋第一個包含至少 6-7 個號碼球的列表
    containers = main_section.find_all(['ul', 'div'], class_=re.compile(r'result|ball|list', re.I))
    
    found_balls = False
    for container in containers:
        # 在容器內找尋球狀元素
        temp_balls = []
        # 尋找內容為數字且類名包含球類特徵的元素
        elements = container.find_all(['li', 'span', 'div'], class_=re.compile(r'ball|no-|num', re.I))
        
        for el in elements:
            txt = el.get_text(strip=True)
            if txt.isdigit():
                val = int(txt)
                if 1 <= val <= 49:
                    temp_balls.append(val)
        
        # 如果這個容器剛好有 7 個數字，且不是日期，那就是我們要的
        if len(temp_balls) >= 7:
            # 過濾掉可能混入的日期數字 (例如 21 號開彩，如果 21 也在球裡面，我們需要確保它不是誤抓日期標籤)
            # 但 21 也可能是中獎號碼，所以我們檢查 class 是否真的是球
            balls = temp_balls[:7]
            found_balls = True
            break

    # 4. 如果結構化定位失敗，使用精確 Regex 備援 (僅限主內容區)
    if not found_balls:
        print("⚠️ 結構定位失敗，啟動精確 Regex 掃描...")
        # 尋找被「+」號連接的 6+1 模式
        full_text = main_section.get_text(separator=' ', strip=True)
        # 排除日期干擾：暫時移除日期字串再搜尋數字
        cleaned_text = full_text.replace(raw_date, "").replace(date_match.group(1).split('-')[-1], "")
        
        nums = re.findall(r'\b\d{1,2}\b', cleaned_text)
        # 過濾出 1-49 的數字並去重
        candidate_nums = []
        for n in nums:
            n_int = int(n)
            if 1 <= n_int <= 49 and n not in date_parts:
                candidate_nums.append(n_int)
        
        # 取前 7 個最像號碼球的數字
        if len(candidate_nums) >= 7:
            balls = candidate_nums[:7]
            found_balls = True

    if found_balls and len(balls) >= 7:
        main = balls[:6]
        extra = balls[6]
        data.update({
            "n1": main[0], "n2": main[1], "n3": main[2],
            "n4": main[3], "n5": main[4], "n6": main[5],
            "extra": extra
        })
        print(f"✅ 解析成功: {main} + {extra}")
    else:
        print(f"❌ 號碼解析不完整。找到的候選數字: {balls}")

    # 獎金資訊
    prize_match = re.search(r'1st.*?HK\$?([\d,]+).*?([\d.]+)', soup.get_text(), re.I | re.DOTALL)
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
            # 匹配格式: /en/mark-six/results/2026-02-21
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
                print(f"ℹ️ 日期 {target_date} 已在 CSV 中，略過更新。")
                # 即使已存在，我們還是可以手動測試，若要強制更新可註解掉 return
                return

        # 3. 抓取並解析
        time.sleep(2) # 延遲確保不被封
        detail_resp = requests.get(latest_link, headers=headers, timeout=20)
        rec = parse_draw_page(detail_resp.text, latest_link)

        if rec.get("n1") and rec.get("extra"):
            new_row = pd.DataFrame([rec])
            # 整合並排序
            df = pd.concat([df, new_row], ignore_index=True)
            df['dt'] = pd.to_datetime(df['date'], format='%Y/%m/%d')
            df = df.sort_values('dt', ascending=True).drop(columns=['dt'])
            df.to_csv(csv_path, index=False)
            print(f"🎉 數據更新成功: {rec['date']} -> {rec['n1']}, {rec['n2']}, {rec['n3']}, {rec['n4']}, {rec['n5']}, {rec['n6']} + {rec['extra']}")
        else:
            print("❌ 號碼解析失敗，未獲取到有效號碼。")
            
    except Exception as e:
        print(f"❌ 發生錯誤: {e}")

if __name__ == "__main__":
    update_marksix()
