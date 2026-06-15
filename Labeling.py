from openai import OpenAI
import json


# OPENAI API を使ってニュースを分類する


#cmdにて　ollama serve で起動
client = OpenAI()
def fix_json(text):
    # 1. 余計なコードブロックを除去
    text = text.strip().strip("`").strip()

    # 2. JSON が文字列として返ってきた場合の処理
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]

    # 3. エスケープされた \" を修正
    text = text.replace('\\"', '"')

    # 4. JSON の前後に余計な文章がある場合を削除
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start:end+1]

    return text

def classify_news(title, text):
    if len(title) > 100:
        title = title[:100]
    if len(text) > 500:
        text = text[:500]

    prompt = f"""
あなたは金融ニュースの専門アナリストです。
以下のニュースタイトル、本文、GDELT の V2Organizations / V2Themes を読み、
次の6分類のいずれかに分類してください。

1. Micro（日本の特定の上場企業に直接関係するニュース）
2. Sector Macro（特定の業界に影響するニュース）
3. Country Macro（日本国内のマクロ経済・政策）
4. Global Macro（世界経済・地政学・国際要因）
5. Mixed（Micro/Macro が混在している、あるいはどちらとも言い難いニュース）
6. Other（上記に当てはまらない、あるいは情報が少なくて判断できないニュース）

補足：
- Organizations に日本企業が含まれていても、内容が業界全体なら Sector Macro。
- Themes に金融政策・為替・地政学が含まれる場合は Macro 系を優先する。
- Organizations に国際機関（UN, IMF, NATO, WHO など）が多い場合は Global Macro。
- 人名（Suzuki, Mori など）は企業と区別する。
- 完璧でなくてよいので、最も妥当なカテゴリを1つ選ぶ。

出力は必ず次のJSON形式で返してください：
{{"label": "...", "reason": "..."}}

ニュースタイトル:
{title}

ニュース本文:
{text}

"""


    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a financial news classification expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            timeout=30
        )

        text = response.choices[0].message.content
    except Exception as e:
        # タイムアウトや接続エラーなどをここでキャッチして、デフォルトのラベルを返す
        errstr = str(e)
        if "timeout" in errstr.lower() or "timed out" in errstr.lower():
            return {"label": "Timeout", "reason": "LLM timeout (30s)", "error": errstr}
        return {"label": "Timeout", "reason": "LLM error", "error": errstr}
    # 余計な \" を除去
    cleaned = fix_json(text)
    try:
        data = json.loads(cleaned)
    except:
        return {"label": "Other", "reason": "LLM parsing error", "raw": text}


    # ★ キーを正規化（"label" → label）
    normalized = {k.strip('"'): v for k, v in data.items()}
    return normalized
