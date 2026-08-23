import pandas as pd

agg = pd.read_csv("measures.csv")
agg = agg[agg["fiscal_year"] <= 2025]

# Firms present in all eight fiscal years
counts = agg.groupby("cik")["fiscal_year"].nunique()
balanced = counts[counts == 8].index
b = agg[agg["cik"].isin(balanced)]

print("Balanced panel firms:", b["cik"].nunique())
print()

t3b = b.groupby("fiscal_year").agg(
    Total=("n_ai", "mean"),
    Deployment=("n_deployment", "mean"),
    Intent=("n_intent", "mean"),
    Boilerplate=("n_boilerplate", "mean"),
    Governance=("n_governance", "mean"),
).round(2)

t3b["Deploy. share"] = (b.groupby("fiscal_year")["share_deployment"].mean() * 100).round(1)
t3b.insert(0, "Firms", b["cik"].nunique())

t3b.to_csv("table3_balanced.csv")
print(t3b.to_string())