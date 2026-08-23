import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS

# Merge text measures with the financial panel
panel = pd.read_csv("panel_final.csv")
meas = pd.read_csv("measures.csv")

df = panel.merge(meas, on=["cik", "fiscal_year"], how="left")

# Firm-years with no AI mentions get zeros
for c in ["n_ai", "n_deployment", "n_intent", "n_boilerplate", "n_governance"]:
    df[c] = df[c].fillna(0)

# Share is undefined when there are no mentions; set to 0
for c in ["share_deployment", "share_intent", "share_boilerplate"]:
    df[c] = df[c].fillna(0)

# Log of counts (log(1+x) handles zeros)
df["log_ai"] = np.log1p(df["n_ai"])
df["log_deploy"] = np.log1p(df["n_deployment"])

# Drop the incomplete year
df = df[df["fiscal_year"] <= 2025]

print("Observations:", len(df))
print("Firms:", df["cik"].nunique())
print()

# Set the panel index: entity, then time
df = df.set_index(["cik", "fiscal_year"])

controls = ["log_assets", "leverage"]


def run(y, x_vars, label):
    """Estimate a panel model with firm and year fixed effects."""
    cols = [y] + x_vars + controls
    d = df[cols].dropna()

    mod = PanelOLS(d[y], d[x_vars + controls],
                   entity_effects=True, time_effects=True)
    res = mod.fit(cov_type="clustered", cluster_entity=True)

    print("===", label)
    print("N =", res.nobs)
    print(res.summary.tables[1])
    print()


# H1: which measure predicts performance?
run("roa", ["log_ai"], "ROA on count of AI mentions")
run("roa", ["share_deployment"], "ROA on share of deployment")
run("roa", ["log_ai", "share_deployment"], "ROA: horse race")

run("sales_growth", ["log_ai"], "Sales growth on count")
run("sales_growth", ["share_deployment"], "Sales growth on share")
run("sales_growth", ["log_ai", "share_deployment"], "Sales growth: horse race")

def run_pooled(y, x_vars, label):
    """Pooled OLS with year effects only, industry controls."""
    from linearmodels.panel import PooledOLS
    cols = [y] + x_vars + controls
    d = df[cols].dropna()
    mod = PanelOLS(d[y], d[x_vars + controls], time_effects=True)
    res = mod.fit(cov_type="clustered", cluster_entity=True)
    print("===", label, "| N =", res.nobs)
    print(res.summary.tables[1])
    print()

run_pooled("roa", ["share_deployment"], "ROA, no firm FE")
run_pooled("sales_growth", ["share_deployment"], "Sales growth, no firm FE")

df = df.reset_index()
df = df.sort_values(["cik", "fiscal_year"])
df["share_deploy_lag1"] = df.groupby("cik")["share_deployment"].shift(1)
df["share_deploy_lag2"] = df.groupby("cik")["share_deployment"].shift(2)
df = df.set_index(["cik", "fiscal_year"])

run("roa", ["share_deploy_lag1"], "ROA on lagged share (t-1)")
run("roa", ["share_deploy_lag2"], "ROA on lagged share (t-2)")

# Does the relationship differ between cohorts?
df = df.reset_index()
early = set(meas[meas["fiscal_year"] <= 2022]["cik"])
df["early"] = df["cik"].isin(early).astype(int)
df["share_x_early"] = df["share_deployment"] * df["early"]
df = df.set_index(["cik", "fiscal_year"])

run("roa", ["share_deployment", "share_x_early"], "ROA: does cohort matter?")