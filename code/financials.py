import pandas as pd
import time
from pipeline import fetch


def get_annual(data, tag, unit="USD"):
    facts = data["facts"].get("us-gaap", {})

    if tag not in facts:
        return {}
    if unit not in facts[tag]["units"]:
        return {}

    result = {}

    for rec in facts[tag]["units"][unit]:
        if rec["form"] != "10-K":
            continue
        if rec.get("fp") != "FY":
            continue

        end = rec["end"]
        if end not in result or rec["filed"] > result[end]["filed"]:
            result[end] = rec

    return {end: r["val"] for end, r in result.items()}


sp500 = pd.read_csv("sp500_cik.csv")

rows = []

for n, row in sp500.iterrows():
    cik = row["cik_str"]
    name = row["Security"]

    url = "https://data.sec.gov/api/xbrl/companyfacts/CIK" + str(cik).zfill(10) + ".json"

    try:
        data = fetch(url).json()
    except Exception as e:
        print(n, name, "Error")
        continue

    assets = get_annual(data, "Assets")
    income = get_annual(data, "NetIncomeLoss")
    rev1 = get_annual(data, "Revenues")
    rev2 = get_annual(data, "RevenueFromContractWithCustomerExcludingAssessedTax")
    liab = get_annual(data, "Liabilities")
    liab2 = get_annual(data, "LiabilitiesAndStockholdersEquity")


    all_dates = set(assets)

    for end in all_dates:
        year = int(end[:4])
        if year < 2017:
            continue

        revenue = rev1.get(end)
        if revenue is None:
            revenue = rev2.get(end)

        rows.append({
            "cik": cik,
            "company": name,
            "period_end": end,
            "fiscal_year": year,
            "assets": assets.get(end),
            "net_income": income.get(end),
            "revenue": revenue,
            "liabilities": liab.get(end) if liab.get(end) is not None else liab2.get(end),
        })

    print(n, "/", len(sp500), name, "|", len(all_dates), "years")
    time.sleep(0.3)

df = pd.DataFrame(rows)
df = df.sort_values(["cik", "period_end"])
df.to_csv("financials.csv", index=False)

print()
print("Strings:", len(df))
print("Companies:", df["cik"].nunique())
print()
print("Filings:")
print(df[["assets", "net_income", "revenue", "liabilities"]].notna().mean().round(3))