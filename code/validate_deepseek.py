import pandas as pd
from openai import OpenAI
import time

client = OpenAI(
    api_key="DEEPSEEK_API_KEYgrep -rn "sk-" .",
    base_url="https://api.deepseek.com"
)

MODEL = "deepseek-chat"

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


df = pd.read_excel("labeling.xlsx")
print("Validating on", len(df), "paragraphs")
print("Model:", MODEL)
print()

results = []

for i, row in df.iterrows():
    try:
        label = classify(row["text"])
    except Exception as e:
        print("ERROR at", row["id"], ":", str(e)[:120])
        label = "error"

    results.append({"id": row["id"], "label_ai": label})

    if (i + 1) % 50 == 0:
        print("Progress:", i + 1, "/", len(df))

    time.sleep(0.2)

pd.DataFrame(results).to_csv("labels_deepseek.csv", index=False)
print()
print("Saved to labels_deepseek.csv")