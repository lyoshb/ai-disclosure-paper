import pandas as pd
import numpy as np

df = pd.read_csv("panel_final.csv")
agg = pd.read_csv("measures.csv")

d = df.merge(agg, on=["cik", "fiscal_year"], how="left")
for c in ["n_ai", "n_deployment", "share_deployment"]:
    d[c] = d[c].fillna(0)
d = d[d["fiscal_year"] <= 2025]

# ---- Table 1: descriptive statistics ----
varlist = {
    "AI paragraphs (count)": "n_ai",
    "Deployment paragraphs (count)": "n_deployment",
    "Deployment share": "share_deployment",
    "Mentions AI (0/1)": "has_ai",
    "Return on assets": "roa",
    "Sales growth": "sales_growth",
    "Log assets": "log_assets",
    "Leverage": "leverage",
}

rows = []
for label, col in varlist.items():
    s = d[col].dropna()
    rows.append({
        "Variable": label,
        "N": len(s),
        "Mean": round(s.mean(), 3),
        "SD": round(s.std(), 3),
        "P25": round(s.quantile(0.25), 3),
        "Median": round(s.median(), 3),
        "P75": round(s.quantile(0.75), 3),
    })

t1 = pd.DataFrame(rows)
t1.to_csv("table1_descriptives.csv", index=False)
print("=== TABLE 1 ===")
print(t1.to_string(index=False))
print()

# ---- Table 3: category composition, pre-2023 adopters ----
early = set(agg[agg["fiscal_year"] <= 2022]["cik"])
e = agg[(agg["cik"].isin(early)) & (agg["fiscal_year"] <= 2025)]

t3 = e.groupby("fiscal_year").agg(
    Firms=("cik", "nunique"),
    Total=("n_ai", "mean"),
    Deployment=("n_deployment", "mean"),
    Intent=("n_intent", "mean"),
    Boilerplate=("n_boilerplate", "mean"),
    Governance=("n_governance", "mean"),
).round(2)

t3["Deploy. share"] = (e.groupby("fiscal_year")["share_deployment"].mean() * 100).round(1)

t3.to_csv("table3_composition.csv")
print("=== TABLE 3 ===")
print(t3.to_string())