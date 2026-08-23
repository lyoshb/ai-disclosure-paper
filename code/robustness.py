import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS

agg = pd.read_csv("measures.csv")
panel = pd.read_csv("panel_final.csv")

early = set(agg[agg["fiscal_year"] <= 2022]["cik"])

# ============================================================
# TEST 1. Placebo: was there a break before 2023?
# ============================================================
print("=" * 60)
print("TEST 1. Placebo — false break points")
print("=" * 60)

d = agg[agg["fiscal_year"] <= 2025].copy()

for cutoff in [2020, 2021, 2023]:
    d["cohort"] = d["cik"].apply(
        lambda c: "early" if c in set(agg[agg["fiscal_year"] < cutoff]["cik"]) else "new"
    )
    post = d[d["fiscal_year"] >= cutoff]
    r = post.groupby("cohort")["share_deployment"].agg(["mean", "size"]).round(3)
    print(f"\nCutoff {cutoff}:")
    print(r)

# ============================================================
# TEST 2. Excluding the technology sector
# ============================================================
print()
print("=" * 60)
print("TEST 2. Excluding Information Technology")
print("=" * 60)

sectors = panel[["cik", "sector"]].drop_duplicates()
d2 = agg.merge(sectors, on="cik", how="left")
d2 = d2[(d2["fiscal_year"] >= 2023) & (d2["fiscal_year"] <= 2025)].copy()
d2["cohort"] = d2["cik"].apply(lambda c: "early" if c in early else "new")

print("\nAll sectors:")
print(d2.groupby("cohort")["share_deployment"].agg(["mean", "size"]).round(3))

no_tech = d2[d2["sector"] != "Information Technology"]
print("\nExcluding IT:")
print(no_tech.groupby("cohort")["share_deployment"].agg(["mean", "size"]).round(3))

print("\nBy sector (post-2022 entrants only):")
new_only = d2[d2["cohort"] == "new"]
print(new_only.groupby("sector")["share_deployment"].agg(["mean", "size"]).round(3))

# ============================================================
# TEST 3. Regressions on the broad measure (with automation)
# ============================================================
print()
print("=" * 60)
print("TEST 3. Robustness of the null result")
print("=" * 60)

df = panel.merge(agg, on=["cik", "fiscal_year"], how="left")
for c in ["n_ai", "n_deployment", "share_deployment"]:
    df[c] = df[c].fillna(0)

df["log_ai"] = np.log1p(df["n_ai"])
df["log_deploy"] = np.log1p(df["n_deployment"])
df = df[df["fiscal_year"] <= 2025].set_index(["cik", "fiscal_year"])

controls = ["log_assets", "leverage"]


def run(y, x_vars, label):
    """Estimate a panel model with firm and year fixed effects."""
    cols = [y] + x_vars + controls
    d = df[cols].dropna()
    res = PanelOLS(d[y], d[x_vars + controls],
                   entity_effects=True, time_effects=True).fit(
        cov_type="clustered", cluster_entity=True)
    print(f"\n--- {label} (N={res.nobs})")
    print(res.summary.tables[1])


# Alternative measure: absolute count of deployment paragraphs
run("roa", ["log_deploy"], "ROA on log count of deployment paragraphs")
run("sales_growth", ["log_deploy"], "Sales growth on log count of deployment")

# Winsorized ROA
df = df.reset_index()
lo, hi = df["roa"].quantile([0.01, 0.99])
df["roa_w"] = df["roa"].clip(lo, hi)
df = df.set_index(["cik", "fiscal_year"])

run("roa_w", ["share_deployment"], "Winsorized ROA on share of deployment")