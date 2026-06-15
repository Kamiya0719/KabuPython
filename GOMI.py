from openai import OpenAI
import json
import pandas as pd
import re


"""

# --- 日本企業辞書のロード ---
path = r"C:\Users\ojiro\Documents\KabuCSharp\KabuCSharp\KabuCSharp\csv\JQuants\SymbolMaster.csv"
df = pd.read_csv(path)
# s33 が 9999 の行を除外
df = df[df["s33"] != 9999]
# 3列目（nameEn）を取得
english_names = df["nameEn"].tolist()

print(english_names[:10])


#print(english_names[:10])


REMOVE_TOKENS = [
    "co", "co.", "ltd", "ltd.", "inc", "inc.", "corporation", "corp", "corp.",
    "company", "holdings", "group", "limited", "plc", "llc", "gmbh", "ag", "sa", "nv", "bv"
]

def normalize_company_name(name):
    if not isinstance(name, str):
        return ""
    name = name.lower()
    name = re.sub(r'[^a-z0-9 ]', ' ', name)
    tokens = name.split()
    tokens = [t for t in tokens if t not in REMOVE_TOKENS]
    return " ".join(tokens)

jp_names_en = set(df["nameEn"].apply(normalize_company_name))
for n in jp_names_en:
    print(f"Detected Japanese company: {n}")

def detect_japanese_companies(organizations):
    detected = []
    for org in organizations:
        n = normalize_company_name(org)
        if n in jp_names_en:
            detected.append(org)
    return detected

"""




