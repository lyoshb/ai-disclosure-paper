import pandas as pd
import os
import time
from openai import OpenAI

client = OpenAI(
    api_key="DEEPSEEK_API_KEY",
    base_url="https://api.deepseek.com"
)

MODEL = "deepseek-chat"
OUT = "labels_full.csv"

VALID = {"deployment", "intent", "boilerplate", "governance", "other"}

PROMPT = """You classify paragraphs from Form 10-K annual reports of U.S. companies that mention artificial intelligence.

Follow this decision procedure IN ORDER. Stop at the first rule that applies.

STEP 1 — Is the paragraph about RULES for AI?
If it mentions any of: AI laws, AI regulation, regulatory guidance on AI, AI compliance obligations, internal AI policies, Responsible AI programs, AI ethics frameworks → governance
This applies EVEN IF the paragraph sits in the risk factors section and uses "may", "could", or "risk".
Do NOT apply for: safe harbor language about forward-looking statements, general mentions of laws unrelated to AI, or data breach risk.

STEP 2 — Is the company describing something it ACTUALLY DOES with AI?
Named products, working systems, specific applications. Verbs: "we use", "we launched", "we deployed", "we have integrated".
→ deployment
Do NOT apply if the company describes what its INDUSTRY does, what CUSTOMERS do, or what it PLANS to do.

STEP 3 — Is the company stating an INTENTION about AI?
Plans, priorities, investments, mission statements. Markers: "our priority is", "we plan to", "we aim", "we are investing in".
→ intent
Key test: if you cannot verify it happened, it is intent.

STEP 4 — Is this legal hedging about RISKS?
Cyber threats, competition, costs, talent, reputation. Includes attackers using AI, deepfakes, safe harbor statements.
→ boilerplate

STEP 5 — Everything else → other
Market or industry trends without company action; the company sells TO firms that adopt AI; sentence fragments.

Reply with ONE word only: governance, deployment, intent, boilerplate, or other."""


def classify(text):
    """Return one category label for a paragraph."""
    r = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        max_tokens=10,
        messages=[
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": text}
        ]
    )
    ans = str(r.choices[0].message.content).strip().lower().rstrip(".")
    return ans if ans in VALID else "unparsed"


# Load paragraphs, keep only those mentioning AI or ML
df = pd.read_csv("paragraphs_all.csv")
t = df["text"].str.lower()
df = df[t.str.contains("artificial intelligence") | t.str.contains("machine learning")]
df = df.reset_index(drop=True)
df["row_id"] = range(len(df))

print("Total paragraphs:", len(df))
print("Model:", MODEL)

# Resume from previous runs
done = set()
if os.path.exists(OUT):
    prev = pd.read_csv(OUT)
    done = set(prev["row_id"])
    print("Already done:", len(done))

todo = df[~df["row_id"].isin(done)]
print("Remaining:", len(todo))
print()

buffer = []
processed = 0
start = time.time()

for i, row in todo.iterrows():
    try:
        label = classify(row["text"])
    except Exception as e:
        print("STOPPED at row", row["row_id"], ":", str(e)[:120])
        break

    buffer.append({"row_id": row["row_id"], "label_ai": label})
    processed += 1

    # Append to file every 100 rows
    if len(buffer) >= 100:
        pd.DataFrame(buffer).to_csv(OUT, mode="a",
                                     header=not os.path.exists(OUT), index=False)
        elapsed = time.time() - start
        left = (len(todo) - processed) / (processed / elapsed) / 60
        print("Saved:", len(done) + processed, "/", len(df),
              "| ETA:", round(left, 1), "min")
        buffer = []

    time.sleep(0.2)

if buffer:
    pd.DataFrame(buffer).to_csv(OUT, mode="a",
                                 header=not os.path.exists(OUT), index=False)

final = pd.read_csv(OUT)
print()
print("Total classified:", len(final))
print(final["label_ai"].value_counts())