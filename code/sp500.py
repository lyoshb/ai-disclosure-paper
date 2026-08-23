import pandas as pd
import requests
from io import StringIO

HEADERS = {"User-Agent": "Alex Balabanov balabanovalex1@gmail.com"}

# 1. Состав S&P 500
url_wiki = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
r = requests.get(url_wiki, headers=HEADERS)
sp500 = pd.read_html(StringIO(r.text))[0]

sp500 = sp500[["Symbol", "Security", "GICS Sector"]]
print("S&P 500:", sp500.shape)
print(sp500.head())
print()

# 2. Справочник SEC
url_sec = "https://www.sec.gov/files/company_tickers.json"
data = requests.get(url_sec, headers=HEADERS).json()

sec = pd.DataFrame.from_dict(data, orient="index")
print("SEC:", sec.shape)
print(sec.head())

merged = sp500.merge(
    sec,
    left_on="Symbol",
    right_on="ticker",
    how="left"
)

print()
print("After Merge:", merged.shape)

# 4. Проверяем потери
missing = merged[merged["cik_str"].isna()]
print("Missing CIK:", len(missing))
print(missing[["Symbol", "Security"]])

# 5. Чистим и сохраняем
result = merged.dropna(subset=["cik_str"])
result = result[["Symbol", "Security", "GICS Sector", "cik_str"]]
result["cik_str"] = result["cik_str"].astype(int)

result.to_csv("sp500_cik.csv", index=False)
print()
print("Saved:", len(result), "Companies")