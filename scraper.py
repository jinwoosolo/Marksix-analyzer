import requests
import pandas as pd
from bs4 import BeautifulSoup
import re

def update_latest():
    url = "https://bet.hkjc.com/en/marksix/results"
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, 'html.parser')
    
    # 實戰：從官網抓最新（實際 selector 需調）
    print("Latest scraped - check manually")
    # TODO: parse latest draw

if __name__ == "__main__":
    update_latest()
