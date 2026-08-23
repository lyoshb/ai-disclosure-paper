import pandas as pd
from pipeline import get_10k_list, extract_from_html, get_html

sp500 = pd.read_csv("sp500_cik.csv")

rows = []

for n, row in sp500.iterrows():
    cik = row["cik_str"]
    name = row["Security"]
    sector = row["GICS Sector"]

    try:
        filings = get_10k_list(cik)
    except Exception as e:
        print(n, name, "Error:", e)
        continue

    for f in filings:
        try:
            html = get_html(f["url"])
            paragraphs = extract_from_html(html)
        except Exception:
            continue

        rows.append({
            "cik": cik,
            "company": name,
            "sector": sector,
            "filing_date": f["date"],
            "n_paragraphs": len(paragraphs)
        })

    print(n, "/", len(sp500), name, "|", len(filings))

df = pd.DataFrame(rows)
df.to_csv("panel_counts.csv", index=False)

print()
print("Strings:", len(df))
print("Companies:", df["cik"].nunique())
print("Have zeros:", (df["n_paragraphs"] == 0).sum())