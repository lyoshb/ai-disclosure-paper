import time
import csv
import os
import hashlib
import pandas as pd

from pipeline import get_10k_list, extract_from_html, fetch

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


sp500 = pd.read_csv("sp500_cik.csv")

print("Companies:", len(sp500))
print()

rows = []
errors = []

for n, row in sp500.iterrows():
    cik = row["cik_str"]
    name = row["Security"]
    sector = row["GICS Sector"]

    try:
        filings = get_10k_list(cik)
    except Exception as e:
        errors.append({"cik": cik, "company": name, "stage": "list", "error": str(e)})
        print(n, "/", len(sp500), name, "| TABLE ERROR")
        continue

    total = 0

    for f in filings:
        try:
            html = get_html(f["url"])
            paragraphs = extract_from_html(html)
        except Exception as e:
            errors.append({"cik": cik, "company": name, "stage": f["date"], "error": str(e)})
            continue

        for j, para in enumerate(paragraphs):
            rows.append({
                "cik": cik,
                "company": name,
                "sector": sector,
                "filing_date": f["date"],
                "paragraph_id": j + 1,
                "n_chars": len(para),
                "text": para
            })

        total += len(paragraphs)

    print(n, "/", len(sp500), name, "|", len(filings), "reports |", total, "columns")

with open("paragraphs_all.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["cik", "company", "sector", "filing_date",
                                       "paragraph_id", "n_chars", "text"])
    w.writeheader()
    w.writerows(rows)

with open("errors.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["cik", "company", "stage", "error"])
    w.writeheader()
    w.writerows(errors)

print()
print("Strings:", len(rows))
print("Errors:", len(errors))