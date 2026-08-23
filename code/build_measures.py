import pandas as pd

# Load paragraphs and rebuild the same row_id used during classification
para = pd.read_csv("paragraphs_all.csv")
t = para["text"].str.lower()
para = para[t.str.contains("artificial intelligence") | t.str.contains("machine learning")]
para = para.reset_index(drop=True)
para["row_id"] = range(len(para))

# Attach model labels
labels = pd.read_csv("labels_full.csv")
para = para.merge(labels, on="row_id", how="left")

print("Paragraphs with labels:", para["label_ai"].notna().sum(), "/", len(para))
print()

# Fiscal year: filings in Jan-Jun refer to the previous year
para["filing_date"] = pd.to_datetime(para["filing_date"])
para["fiscal_year"] = para["filing_date"].dt.year
para.loc[para["filing_date"].dt.month <= 6, "fiscal_year"] -= 1

# Aggregate to firm-year level
agg = para.groupby(["cik", "fiscal_year"]).agg(
    n_ai=("label_ai", "size"),
    n_deployment=("label_ai", lambda s: (s == "deployment").sum()),
    n_intent=("label_ai", lambda s: (s == "intent").sum()),
    n_boilerplate=("label_ai", lambda s: (s == "boilerplate").sum()),
    n_governance=("label_ai", lambda s: (s == "governance").sum()),
).reset_index()

# Shares
agg["share_deployment"] = agg["n_deployment"] / agg["n_ai"]
agg["share_intent"] = agg["n_intent"] / agg["n_ai"]
agg["share_boilerplate"] = agg["n_boilerplate"] / agg["n_ai"]

agg.to_csv("measures.csv", index=False)

print("Firm-years with AI mentions:", len(agg))
print()

# H2 test: did the SHARE of deployment rise as fast as the COUNT?
by_year = agg[agg["fiscal_year"] <= 2025].groupby("fiscal_year").agg(
    firms=("cik", "nunique"),
    mean_ai=("n_ai", "mean"),
    mean_deploy=("n_deployment", "mean"),
    share_deploy=("share_deployment", "mean")
).round(3)

print("H2: counts vs shares by year")
print(by_year)

# Balanced panel: firms present in every year 2018-2025
counts = agg[agg["fiscal_year"] <= 2025].groupby("cik")["fiscal_year"].nunique()
always = counts[counts == 8].index

bal = agg[agg["cik"].isin(always) & (agg["fiscal_year"] <= 2025)]

print()
print("Balanced panel:", bal["cik"].nunique(), "firms")
print(bal.groupby("fiscal_year").agg(
    mean_ai=("n_ai", "mean"),
    mean_deploy=("n_deployment", "mean"),
    share_deploy=("share_deployment", "mean")
).round(3))

# Firms that mentioned AI before 2023 vs those that joined after
early = set(agg[agg["fiscal_year"] <= 2022]["cik"])

recent = agg[(agg["fiscal_year"] >= 2023) & (agg["fiscal_year"] <= 2025)].copy()
recent["cohort"] = recent["cik"].apply(lambda c: "early" if c in early else "new")

print()
print("Post-2022 filings, by cohort:")
print(recent.groupby("cohort").agg(
    firms=("cik", "nunique"),
    obs=("cik", "size"),
    mean_ai=("n_ai", "mean"),
    mean_deploy=("n_deployment", "mean"),
    share_deploy=("share_deployment", "mean")
).round(3))

import matplotlib.pyplot as plt

# Share of deployment mentions by cohort, over time
early = set(agg[agg["fiscal_year"] <= 2022]["cik"])
d = agg[agg["fiscal_year"] <= 2025].copy()
d["cohort"] = d["cik"].apply(lambda c: "Pre-2023 adopters" if c in early else "Post-2022 entrants")

pivot = d.groupby(["fiscal_year", "cohort"])["share_deployment"].mean().unstack() * 100

plt.figure(figsize=(8, 5))
for col in pivot.columns:
    plt.plot(pivot.index, pivot[col], marker="o", label=col)

plt.axvline(2022.5, linestyle="--", color="grey", linewidth=1)
plt.xlabel("Fiscal year")
plt.ylabel("Share of AI mentions describing deployment, %")
plt.title("Figure 2. Talk vs. deployment, by adoption cohort")
plt.legend(loc="upper right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fig2_cohorts.png", dpi=300)
plt.show()

# Absolute counts, not shares, for pre-2023 adopters
early_only = d[d["cohort"] == "Pre-2023 adopters"]
print(early_only.groupby("fiscal_year").agg(
    mean_ai=("n_ai", "mean"),
    mean_deploy=("n_deployment", "mean"),
    mean_boiler=("n_boilerplate", "mean"),
    mean_gov=("n_governance", "mean")
).round(2))

# Absolute counts by category, pre-2023 adopters
e = d[d["cohort"] == "Pre-2023 adopters"].groupby("fiscal_year").agg(
    deployment=("n_deployment", "mean"),
    boilerplate=("n_boilerplate", "mean"),
    governance=("n_governance", "mean")
)

plt.figure(figsize=(8, 5))
for col in e.columns:
    plt.plot(e.index, e[col], marker="o", label=col)

plt.axvline(2022.5, linestyle="--", color="grey", linewidth=1)
plt.xlabel("Fiscal year")
plt.ylabel("Mean paragraphs per firm-year")
plt.title("Figure 3. What grew after ChatGPT: risk language, not deployment")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fig3_categories.png", dpi=300)
plt.show()

