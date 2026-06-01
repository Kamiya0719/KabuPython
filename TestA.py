import sys
import time
import traceback
from datetime import date, timedelta
from pathlib import Path

import tdnet
import pandas as pd
from tdnet import CK, extract_values, extracted_to_dict

# ----------------------------------------
# 1. 抽出する財務指標（あなたの用途に最適化）
# ----------------------------------------
FIN_KEYS = [
    CK.REVENUE,                 # 売上高
    CK.GROSS_PROFIT,            # 売上総利益
    CK.OPERATING_INCOME,        # 営業利益
    CK.ORDINARY_INCOME,         # 経常利益
    CK.NET_INCOME_PARENT,       # 親会社株主に帰属する純利益
    CK.EPS,                     # EPS
    CK.TOTAL_ASSETS,            # 総資産
    CK.SHAREHOLDERS_EQUITY,     # 自己資本
    CK.CASH_AND_EQUIVALENTS,    # 現金同等物
    CK.INTEREST_BEARING_DEBT,   # 有利子負債
    CK.OPERATING_CF,            # 営業CF
    CK.INVESTING_CF,            # 投資CF
    CK.FINANCING_CF,            # 財務CF
    CK.ROE,                     # ROE
    CK.ROA,                     # ROA
]

# ----------------------------------------
# 2. XBRL → 財務特徴量抽出
# ----------------------------------------
def extract_xbrl_features(filing, log_root: Path | None = None):
    try:
        stmts = retry_on_404(
            filing.xbrl,
            retries=3,
            wait_seconds=1.0,
            log_root=log_root,
            description=f"filing.xbrl for {filing.company_code} {filing.title}",
        )
    except Exception as e:
        msg = f"[XBRL ERROR] {filing.company_code} {filing.title}: {type(e).__name__}: {e}"
        print(msg)
        traceback.print_exc()
        if log_root is not None:
            log_error(msg, log_root)
        return None

    # メタ情報
    # Statements オブジェクト自体に fiscal_period/consolidated 属性はないため、
    # income_statement / balance_sheet / cash_flow_statement から検出する。
    period = None
    consolidated = True
    for getter in (stmts.income_statement, stmts.balance_sheet, stmts.cash_flow_statement):
        try:
            fs = getter()
        except Exception:
            continue
        if fs is None:
            continue
        if period is None and getattr(fs, "period", None) is not None:
            period = str(fs.period)
        consolidated = getattr(fs, "consolidated", True)
        if period is not None:
            break

    meta = {
        "code": filing.company_code,
        "company_name": filing.company_name,
        "title": filing.title,
        "pubdate": filing.pubdate,
        "fiscal_period": period,
        "consolidated": consolidated,
    }

    # 財務データ（current）
    try:
        fin = extract_values(
            stmts,
            FIN_KEYS,
            period="current",
            consolidated=True
        )
        fin = extracted_to_dict(fin)
    except Exception as e:
        print(f"[EXTRACT ERROR] {filing.company_code}: {e}")
        fin = {}

    # 結合
    return {**meta, **fin}

# ----------------------------------------
# 3. 1日分の開示を処理
# 404 エラー判定
def is_404_error(exception: Exception) -> bool:
    if hasattr(exception, "code") and getattr(exception, "code") == 404:
        return True
    if hasattr(exception, "status_code") and getattr(exception, "status_code") == 404:
        return True
    message = str(exception)
    if "404" in message and "HTTP" in message:
        return True
    return False


# 404 の場合に最大 3 回まで再試行する汎用ヘルパー
def retry_on_404(
    func,
    *args,
    retries: int = 3,
    wait_seconds: float = 1.0,
    log_root: Path | None = None,
    description: str | None = None,
    **kwargs,
):
    description = description or getattr(func, "__name__", "operation")
    for attempt in range(1, retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if not is_404_error(e):
                raise
            msg = f"[RETRY {attempt}/{retries}] {description} failed with 404: {e}"
            print(msg)
            traceback.print_exc()
            if log_root is not None:
                log_error(msg, log_root)
            if attempt == retries:
                raise
            time.sleep(wait_seconds)
# ----------------------------------------
# 指定日付の TDnet 開示を取得し、抽出結果を DataFrame にまとめる
def process_tdnet(date_str, log_root: Path | None = None):
    meta_cols = ["code", "company_name", "title", "pubdate", "fiscal_period", "consolidated"]
    try:
        filings = retry_on_404(
            tdnet.documents,
            date_str,
            has_xbrl=True,
            retries=3,
            wait_seconds=1.0,
            log_root=log_root,
            description=f"tdnet.documents for {date_str}",
        )
    except Exception as e:
        msg = f"[TDNET ERROR] date={date_str} type={type(e).__name__} error={e}"
        print(msg)
        traceback.print_exc()
        if log_root is not None:
            log_error(msg, log_root)
        return pd.DataFrame(columns=meta_cols)

    rows = []

    for f in filings:
        row = extract_xbrl_features(f, log_root=log_root)
        if row is not None:
            rows.append(row)

    df = pd.DataFrame(rows)

    # 列順を整える（見やすさのため）
    meta_cols = ["code", "company_name", "title", "pubdate", "fiscal_period", "consolidated"]
    if df.empty:
        return pd.DataFrame(columns=meta_cols)

    fin_cols = [col for col in df.columns if col not in meta_cols]
    df = df[meta_cols + fin_cols]

    return df

# ----------------------------------------
# 4. 日付ループユーティリティ
# 指定年の全日付を YYYYMMDD 形式の文字列で生成する

def get_date_strings_for_year(year):
    current = date(year, 1, 1)
    end = date(year, 12, 31)
    while current <= end:
        yield current.strftime("%Y%m%d")
        current += timedelta(days=1)


# ----------------------------------------
# 5. 年単位の処理
# エラーを年別ログファイルに書き込む

# エラーを年別ログファイルに書き込む
def log_error(message: str, log_root: Path) -> None:
    log_root.mkdir(parents=True, exist_ok=True)
    log_path = log_root / "process.log"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(message + "\n")


# 年単位で全日付を処理し、日別 CSV とログを出力する
def process_year(year: int, output_root: Path | str = None):
    if output_root is None:
        output_root = Path(__file__).resolve().parent / "csv"
    output_root = Path(output_root)
    year_dir = output_root / str(year)
    year_dir.mkdir(parents=True, exist_ok=True)
    log_root = year_dir

    for date_str in get_date_strings_for_year(year):
        print(f"Processing {date_str}...")
        try:
            df = process_tdnet(date_str, log_root=log_root)
            output_path = year_dir / f"my_features_{date_str}.csv"
            save_to_csv(df, output_path)
            print(f"  saved {output_path}")
        except Exception as e:
            msg = f"[ERROR] {date_str}: {type(e).__name__}: {e}"
            print(msg)
            traceback.print_exc()
            log_error(msg, log_root)


# ----------------------------------------
# 6. CSV 保存ユーティリティ
# DataFrame を指定したパスに CSV として保存する
# ----------------------------------------
def save_to_csv(df: pd.DataFrame, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

# ----------------------------------------
# 5. 実行例
# メイン実行部: 年を指定して処理を開始する
# ----------------------------------------
if __name__ == "__main__":
    #year = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
    year = 2024
    output_root = Path(__file__).resolve().parent / "csv"
    process_year(year, output_root)

    
# J-Quants 株価データを読み込み（既に取得済み）
#df_price = pd.read_csv("jquants_price.csv")
# 翌営業日をマージ
#df = df_xbrl.merge(df_price, on=["code", "date"], how="left")
# リターン計算
#df["ret_1d"] = (df["close_t1"] - df["close_t"]) / df["close_t"]
#df["ret_10d"] = (df["close_t10"] - df["close_t"]) / df["close_t"]
# S3 にアップロード
#df.to_csv("s3://your-bucket/xbrl_features/20260501.csv", index=False)




"""
✔ PL（損益計算書）
売上高（CK.REVENUE）
営業利益（CK.OPERATING_INCOME）
経常利益（CK.ORDINARY_INCOME）
親会社株主に帰属する純利益（CK.NET_INCOME_PARENT）
EPS（CK.EPS）
売上総利益（CK.GROSS_PROFIT）
営業利益率（営業利益 / 売上高）
純利益率（純利益 / 売上高）

✔ BS（貸借対照表）
総資産（CK.TOTAL_ASSETS）
自己資本（CK.EQUITY）
現金及び現金同等物（CK.CASH_EQUIVALENTS）
有利子負債（CK.INTEREST_BEARING_DEBT）
流動比率（流動資産 / 流動負債）

✔ CF（キャッシュフロー）
営業CF（CK.CF_OPERATING）
投資CF（CK.CF_INVESTING）
財務CF（CK.CF_FINANCING）
フリーCF（営業CF + 投資CF）

✔ KPI（企業指標）
ROE（CK.ROE）
ROA（CK.ROA）
営業CFマージン（営業CF / 売上高）

✔ メタ情報（重要）
決算期（fiscal_period）
連結/単体（consolidated）
発表日（pubdate）
対象期間（current / previous）
"""