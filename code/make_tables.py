import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS

panel = pd.read_csv("panel_final.csv")
agg = pd.read_csv("measures.csv")

df = panel.merge(agg, on=["cik", "fiscal_year"], how="left")
for c in ["n_ai", "n_deployment", "share_deployment"]:
    df[c] = df[c].fillna(0)

df["log_ai"] = np.log1p(df["n_ai"])
df["log_deploy"] = np.log1p(df["n_deployment"])
df = df[df["fiscal_year"] <= 2025]

# Lags
df = df.sort_values(["cik", "fiscal_year"])
df["share_lag1"] = df.groupby("cik")["share_deployment"].shift(1)
df["share_lag2"] = df.groupby("cik")["share_deployment"].shift(2)

early = set(agg[agg["fiscal_year"] <= 2022]["cik"])
df["early"] = df["cik"].isin(early).astype(int)
df["share_x_early"] = df["share_deployment"] * df["early"]

df = df.set_index(["cik", "fiscal_year"])

controls = ["log_assets", "leverage"]


def stars(p):
    """Significance stars."""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def estimate(y, x_vars, firm_fe=True):
    """Run one model, return a dict of formatted results."""
    cols = [y] + x_vars + controls
    d = df[cols].dropna()
    res = PanelOLS(d[y], d[x_vars + controls],
                   entity_effects=firm_fe, time_effects=True).fit(
        cov_type="clustered", cluster_entity=True)

    out = {}
    for v in x_vars + controls:
        b = res.params[v]
        se = res.std_errors[v]
        p = res.pvalues[v]
        out[v] = f"{b:.4f}{stars(p)}\n({se:.4f})"
    out["N"] = int(res.nobs)
    out["R2 within"] = round(res.rsquared_within, 3)
    return out


# ---- Table 4: main specifications ----
models4 = {
    "(1) ROA": estimate("roa", ["log_ai"]),
    "(2) ROA": estimate("roa", ["share_deployment"]),
    "(3) ROA": estimate("roa", ["log_ai", "share_deployment"]),
    "(4) Sales gr.": estimate("sales_growth", ["log_ai"]),
    "(5) Sales gr.": estimate("sales_growth", ["share_deployment"]),
    "(6) Sales gr.": estimate("sales_growth", ["log_ai", "share_deployment"]),
}

t4 = pd.DataFrame(models4)
t4.to_csv("table4_main.csv")
print("=== TABLE 4: Main specifications ===")
print(t4.to_string())
print()

# ---- Table 5: robustness ----
models5 = {
    "(1) No firm FE": estimate("roa", ["share_deployment"], firm_fe=False),
    "(2) Lag 1": estimate("roa", ["share_lag1"]),
    "(3) Lag 2": estimate("roa", ["share_lag2"]),
    "(4) Cohort int.": estimate("roa", ["share_deployment", "share_x_early"]),
    "(5) Count deploy": estimate("roa", ["log_deploy"]),
}

t5 = pd.DataFrame(models5)
t5.to_csv("table5_robustness.csv")
print("=== TABLE 5: Robustness ===")
print(t5.to_string())

# Fix row order: variables of interest first, then controls, then stats
order4 = ["log_ai", "share_deployment", "log_assets", "leverage", "N", "R2 within"]
t4 = t4.reindex(order4)
t4.to_csv("table4_main.csv")

order5 = ["share_deployment", "share_lag1", "share_lag2", "share_x_early",
          "log_deploy", "log_assets", "leverage", "N", "R2 within"]
t5 = t5.reindex(order5)
t5.to_csv("table5_robustness.csv")

print("Reordered and saved")