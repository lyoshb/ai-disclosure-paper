import pandas as pd

panel = pd.read_csv("panel_counts.csv")
fin = pd.read_csv("financials.csv")

print("Panel:", panel.shape)
print("Finance:", fin.shape)
print()

panel["filing_date"] = pd.to_datetime(panel["filing_date"])
panel["year"] = panel["filing_date"].dt.year
panel["month"] = panel["filing_date"].dt.month

panel["fiscal_year"] = panel["year"]
panel.loc[panel["month"] <= 6, "fiscal_year"] = panel["year"] - 1

print("Test on example:")
check = panel[panel["company"].isin(["Apple Inc.", "Coca-Cola Company (The)"])]
print(check[["company", "filing_date", "fiscal_year", "n_paragraphs"]].head(20))

fin["cik"] = fin["cik"].astype(int)
panel["cik"] = panel["cik"].astype(int)

fin = fin.sort_values("period_end")
fin = fin.drop_duplicates(subset=["cik", "fiscal_year"], keep="last")

panel = panel.drop_duplicates(subset=["cik", "fiscal_year"], keep="first")

print("Posle dedup - panel:", panel.shape, "fin:", fin.shape)

df = panel.merge(
    fin[["cik", "fiscal_year", "assets", "net_income", "revenue", "liabilities"]],
    on=["cik", "fiscal_year"],
    how="left"
)

print()
print("After merge:", df.shape)
print()
print("Completeness:")
print(df[["assets", "net_income", "revenue"]].notna().mean().round(3))
print()

missing = df[df["assets"].isna()]
print("No finance:", len(missing))
print(missing["company"].value_counts().head(10))

dups = df[df.duplicated(subset=["cik", "fiscal_year"], keep=False)]
print("Same strings:", len(dups))
print()
print(dups[["company", "filing_date", "fiscal_year", "assets", "revenue"]].head(20))


df["roa"] = df["net_income"] / df["assets"]

import numpy as np
df["log_assets"] = np.log(df["assets"].where(df["assets"] > 0))
df["leverage"] = df["liabilities"] / df["assets"]

df = df.sort_values(["cik", "fiscal_year"])
df["revenue_prev"] = df.groupby("cik")["revenue"].shift(1)
df["sales_growth"] = df["revenue"] / df["revenue_prev"] - 1
df["sales_growth"] = df["sales_growth"].replace([np.inf, -np.inf], np.nan)
df.loc[df["sales_growth"] > 5, "sales_growth"] = np.nan
df.loc[df["sales_growth"] < -0.9, "sales_growth"] = np.nan

df["has_ai"] = (df["n_paragraphs"] > 0).astype(int)

df.to_csv("panel_final.csv", index=False)

print()
print("Result panel:", df.shape)
print()
print(df[["n_paragraphs", "roa", "sales_growth", "leverage", "has_ai"]].describe().round(3))