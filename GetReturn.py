import tdnet
import requests

# filing オブジェクトから fetch_pdf のシグネチャを確認
filing = tdnet.documents("20260501", has_xbrl=True)[0]

# help(filing.fetch_pdf) で引数を確認
print(help(filing.fetch_pdf))

# または、requests で直接取得試行
print(f"document_url: {filing.document_url}")
print(f"xbrl_url: {filing.xbrl_url}")

# User-Agent付きで直接リクエスト試行
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}
try:
    r = requests.get(filing.document_url, headers=headers, allow_redirects=True, timeout=10)
    print(f"Status: {r.status_code}, Content-Type: {r.headers.get('Content-Type')}")
except Exception as e:
    print(f"Error: {e}")

"""

終値（Close）
始値（Open）
高値（High）
安値（Low）
出来高（Volume）


ラベル（Return）
翌日リターン
10日後リターン

方向ラベル（分類用）
Positive（> +1%）
Neutral（-1%〜+1%）
Negative（< -1%）
"""



"""
純利益
eps	EPS
total_assets	総資産
equity	自己資本
cash_eq	現金同等物
cf_operating	営業CF
roe	ROE
roa	ROA
…	他の財務指標
close_t	発表日の翌営業日の終値
ret_1d	翌日リターン
ret_10d	10日後リターン
label_1d	翌日方向ラベル
label_10d	10日後方向ラベル

"""

"""
| column | 説明 |
| --- | --- |
| code | 銘柄コード |
| pubdate | 開示日 |
| fiscal_period | 決算期 |
| consolidated | 連結/単体 |
| revenue | 売上高 |
| operating_income | 営業利益 |
| net_income | 純利益 |
| eps | EPS |
| total_assets | 総資産 |
| equity | 自己資本 |
| cash_eq | 現金同等物 |
| cf_operating | 営業CF |
| roe | ROE |
| roa | ROA |
| … | 他の財務指標 |
| close_t | 発表日の翌営業日の終値 |
| ret_1d | 翌日リターン |
| ret_10d | 10日後リターン |
| label_1d | 翌日方向ラベル |
| label_10d | 10日後方向ラベル |
"""

