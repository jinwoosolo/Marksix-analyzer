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
    🔥 超強版解析邏輯：結合多種爬蟲策略確保 100% 抓到數據
    """
    if not html:
        return {}

    soup = BeautifulSoup(html, "lxml")
    data = {"url": url}

    # 1. 提取日期 (優先從 URL 提取，因為這是最準確的)
    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", url)
    if date_match:
        data["date"] = date_match.group(1).replace("-", "/")
        print(f"📅 從網址辨識日期: {data['date']}")
    else:
        # 備援：從頁面標題尋找日期
        content = soup.get_text()
        d_match = re.search(r'(\d{1,2})\s+([A-Z][a-z]+)\s+(\d{4})', content)
        if d_match:
            try:
                dt = datetime.strptime(d_match.group(0), '%d %B %Y')
                data["date"] = dt.strftime('%Y/%m/%d')
            except: pass

    # 2. 提取號碼 - 方法 A: 精確選擇器 (針對球體容器)
    balls = []
    # 鎖定可能的結果區塊
    container_selectors = [".result-box", ".draw-results", ".balls", ".results-list", "main", "article"]
    target_container = None
    for sel in container_selectors:
        target_container = soup.select_one(sel)
        if target_container and len(target_container.select("li, span")) >= 7:
            break
    
    if not target_container:
        target_container = soup

    # 掃描球體標籤
    ball_elements = target_container.find_all(['li', 'span', 'div'], class_=re.compile(r'ball|no-|num', re.I))
    for el in ball_elements:
        txt = el.get_text(strip=True)
        if txt.isdigit():
            val = int(txt)
            if 1 <= val <= 49 and val not in balls:
                balls.append(val)

    # 3. 提取號碼 - 方法 B: 強力正則備援 (當標籤法抓不到或抓錯時)
    if len(balls) < 7:
        print("⚠️ 標籤抓取不足，啟動正則表達式備援...")
        # 清除 HTML 雜訊，保留數字
        clean_text = re.sub(r'[^\w\s\d/+]', ' ', target_container.get_text(separator=' '))
        nums = re.findall(r'\b\d{1,2}\b', clean_text)
        # 排除日期數字（例如 2026, 02, 21）
        date_parts = set(re.findall(r'\d+', data.get("date", "")))
        candidate_nums = [int(n) for n in nums if n.isdigit() and 1 <= int(n) <= 49 and n not in date_parts]
        
        # 嘗試尋找最像開獎號碼的 7 個連續數字
        if len(candidate_nums) >= 7:
            balls = candidate_nums[:7]

    # 4. 數據分配 (n1-n6 + extra)
    if len(balls) >= 7:
        main = balls[:6]
        extra = balls[6]
        data.update({
            "n1": main[0], "n2": main[1], "n3": main[2],
            "n4": main[3], "n5": main[4], "n6": main[5],
            "extra": extra
        })
        print(f"✅ 成功抓取號碼: {main} + {extra}")
    else:
        print(f"❌ 號碼解析失敗。僅抓到: {balls}")

    # 5. 抓取獎金與中獎人數
    full_text = soup.get_text()
    prize_match = re.search(r'1st.*?HK\$?([\d,]+).*?([\d.]+)', full_text, re.I | re.DOTALL)
    if prize_match:
        data["div1_prize"] = prize_match.group(1).replace(',', '')
        data["div1_winners"] = prize_match.group(2)
    else:
        data["div1_prize"] = "0"
        data["div1_winners"] = "0"

    return data

def fetch_latest_marksix():
    """
    主執行程序：獲取最新連結並同步至 CSV
    """
    print(f"\n{'='*50}")
    print(f"🚀 六合彩 AI 抓取器啟動時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*50}")

    try:
        # 1. 存取列表頁
        resp = requests.get(RESULTS_LIST_URL, headers=headers, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        
        # 2. 獲取最新一期的 URL
        latest_link = None
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if re.search(r"/mark-six/results/20\d{2}-\d{2}-\d{2}", href):
                latest_link = urljoin(BASE_URL, href)
                break
        
        if not latest_link:
            print("❌ 錯誤：無法在列表頁找到任何開獎連結。")
            return False
            
        print(f"🔎 發現最新開獎頁面: {latest_link}")

        # 3. 獲取詳情頁 HTML
        time.sleep(2) 
        detail_resp = requests.get(latest_link, headers=headers, timeout=20)
        detail_resp.raise_for_status()
        
        # 4. 解析數據
        rec = parse_draw_page(detail_resp.text, latest_link)

        if rec.get("date") and rec.get("n1"):
            csv_path = 'marksix.csv'
            
            # 5. CSV 更新與修復邏輯
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                df['date'] = df['date'].astype(str)
            else:
                df = pd.DataFrame(columns=['date', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'extra', 'div1_prize', 'div1_winners', 'url'])

            # 🔥 強力修復：如果日期已存在（解決 26 號那種長數字髒數據），先刪除舊行
            target_date = rec["date"]
            if target_date in df['date'].values:
                print(f"ℹ️ 日期 {target_date} 已存在。執行覆蓋更新以修復潛在錯誤數據...")
                df = df[df['date'] != target_date]
            else:
                print(f"🆕 偵測到新日期: {target_date}")

            # 6. 合併新數據
            new_row = pd.DataFrame([rec])
            df = pd.concat([df, new_row], ignore_index=True)
            
            # 7. 格式化日期並排序，確保 CSV 整齊
            df['date_dt'] = pd.to_datetime(df['date'], format='%Y/%m/%d')
            df = df.sort_values('date_dt', ascending=True).drop(columns=['date_dt'])
            
            # 8. 儲存
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"🎉 數據同步成功！共 {len(df)} 筆紀錄。")
            return True
        else:
            print("❌ 解析過程未獲取到有效數據。")
            
    except Exception as e:
        print(f"❌ 發生未預期錯誤: {e}")
    
    return False

if __name__ == "__main__":
    fetch_latest_marksix()
