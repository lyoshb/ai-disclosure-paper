import requests
import time
import warnings
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

HEADERS = {"User-Agent": "Alex Balabanov balabanovalex1@gmail.com"}
KEYWORDS = ["artificial intelligence", "machine learning", "automation"]


def fetch(url, tries=3):
    for attempt in range(tries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            if r.status_code == 200:
                return r
            print("      status", r.status_code, "attempt", attempt + 1)
        except Exception:
            print("      network error, attempt", attempt + 1)
        time.sleep(2 * (attempt + 1))
    raise Exception("Couldnt do after " + str(tries) + " tries: " + url)


def get_10k_list(cik, min_year=2019):
    cik_padded = str(cik).zfill(10)
    url = "https://data.sec.gov/submissions/CIK" + cik_padded + ".json"

    data = fetch(url).json()
    company = data["name"]

    blocks = [data["filings"]["recent"]]

    for f in data["filings"].get("files", []):
        if f["filingTo"] >= str(min_year):
            arch_url = "https://data.sec.gov/submissions/" + f["name"]
            blocks.append(fetch(arch_url).json())
            time.sleep(0.3)

    result = []

    for block in blocks:
        for i in range(len(block["form"])):
            if block["form"][i] != "10-K":
                continue

            date = block["filingDate"][i]
            if date < str(min_year):
                continue

            accession = block["accessionNumber"][i].replace("-", "")
            doc = block["primaryDocument"][i]
            link = ("https://www.sec.gov/Archives/edgar/data/"
                    + str(cik) + "/" + accession + "/" + doc)

            result.append({
                "cik": cik,
                "company": company,
                "date": date,
                "url": link
            })

    return result


def extract_from_html(html):
    soup = BeautifulSoup(html, "lxml")
    blocks = soup.find_all(["p", "div", "span"])

    found = []
    seen = set()

    for block in blocks:
        text = block.get_text(separator=" ")
        text = " ".join(text.split())

        if len(text) < 100 or len(text) > 6000:
            continue
        if text in seen:
            continue

        text_lower = text.lower()
        for kw in KEYWORDS:
            if kw in text_lower:
                found.append(text)
                seen.add(text)
                break

    return found

import os
import hashlib

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def get_html(url):
    name = hashlib.md5(url.encode()).hexdigest() + ".html"
    path = os.path.join(CACHE_DIR, name)

    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    html = fetch(url).text

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)

    time.sleep(0.5)
    return html