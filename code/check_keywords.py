import pandas as pd

df = pd.read_csv("paragraphs_all.csv")
t = df["text"].str.lower()

df["has_ai"] = t.str.contains("artificial intelligence")
df["has_ml"] = t.str.contains("machine learning")
df["has_auto"] = t.str.contains("automation")

only_auto = df[df["has_auto"] & ~df["has_ai"] & ~df["has_ml"]]

print("Total columns:", len(df))
print("Only Automation:", len(only_auto), "(", round(100*len(only_auto)/len(df), 1), "%)")
print()
print(only_auto["sector"].value_counts())

sample = only_auto.sample(10, random_state=42)
for i, r in sample.iterrows():
    print(r["company"], "|", r["sector"])
    print(r["text"][:300])
    print("---")

