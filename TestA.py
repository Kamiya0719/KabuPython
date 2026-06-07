import sys
import time
import traceback
from datetime import date, timedelta
from pathlib import Path
import io
import zipfile
from html.parser import HTMLParser

import tdnet
import pandas as pd
import fitz
import requests
from janome.tokenizer import Tokenizer
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
            retries=1,
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
            raise
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

    # 財務データは当期と前期の両方を取得しておく
    try:
        # 期間フィルタ無しで全 CK を取得し、当期の値を優先して保存
        fin_all = extract_values(
            stmts,
            period=None,
            consolidated=True,
        )
        fin = extracted_to_dict(fin_all)

        # 前期だけを別途取得し、prior サフィックス付きで保存
        fin_prior = extracted_to_dict(
            extract_values(
                stmts,
                period="prior",
                consolidated=True,
            )
        )
        for key, value in fin_prior.items():
            if value is not None:
                fin[f"{key}_prior"] = value
    except Exception as e:
        print(f"[EXTRACT ERROR] {filing.company_code}: {e}")
        raise

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
    retries: int = 1,
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
            retries=1,
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
        raise


    rows = []

    for f in filings:
        row = extract_xbrl_features(f, log_root=log_root)
        if row is not None:
            rows.append(row)

    df = pd.DataFrame(rows)

    # 列順を整える（見やすさのため）
    meta_cols = ["code", "company_name", "title", "pubdate", "fiscal_period", "consolidated"]
    if df.empty:
        return pd.DataFrame(columns=meta_cols), filings

    fin_cols = [col for col in df.columns if col not in meta_cols]
    df = df[meta_cols + fin_cols]

    return df, filings

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
            df, filings = process_tdnet(date_str, log_root=log_root)
            output_path = year_dir / f"my_features_{date_str}.csv"
            if df.empty:
                print(f"  no XBRL data for {date_str}, skip saving")
            else:
                save_to_csv(df, output_path)
                print(f"  saved {output_path}")

            process_pdf_files(
                filings,
                year=year,
                date_str=date_str,
                base_root=output_root,
                log_root=log_root,
            )
        except Exception as e:
            msg = f"[ERROR] {date_str}: {type(e).__name__}: {e}"
            print(msg)
            traceback.print_exc()
            log_error(msg, log_root)
            raise


# ----------------------------------------
# 6. CSV 保存ユーティリティ
# DataFrame を指定したパスに CSV として保存する
# ----------------------------------------
def save_to_csv(df: pd.DataFrame, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def filter_features_by_keys(input_csv: str | Path, output_csv: str | Path, keys: list[str] | None = None) -> None:
    """
    既に保存された特徴量CSVから、メタ情報 + 指定の財務指標だけを抽出して保存するユーティリティ。
    keys を None にすると `FIN_KEYS` を使用します。
    """
    keys = keys or FIN_KEYS
    keys = [str(k) for k in keys]
    meta_cols = ["code", "company_name", "title", "pubdate", "fiscal_period", "consolidated"]
    df = pd.read_csv(input_csv)
    # 存在する列のみを選択。prior 列も含める。
    selected = []
    for c in meta_cols:
        if c in df.columns:
            selected.append(c)
    for key in keys:
        if key in df.columns:
            selected.append(key)
        prior_key = f"{key}_prior"
        if prior_key in df.columns:
            selected.append(prior_key)
    out_df = df[selected]
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)


def extract_text_from_saved_pdfs(meta_csv: str | Path, base_root: str | Path | None = None) -> None:
    """
    保存済みのPDFメタデータCSVから、各PDFファイルのテキストを抽出して保存する。
    base_root が None の場合、meta_csv と同じ csv フォルダを基準とします。
    """
    meta_path = Path(meta_csv)
    if base_root is None:
        base_root = meta_path.parent.parent.parent
    base_root = Path(base_root)
    
    df_meta = pd.read_csv(meta_csv)
    text_dir = base_root / "pdf_text"
    text_rows = []
    
    print(f"extract_text_from_saved_pdfs: processing {len(df_meta)} PDFs...")
    for idx, row in df_meta.iterrows():
        pdf_path = Path(row["pdf_path"])
        if not pdf_path.exists():
            print(f"  [WARN] PDF not found: {pdf_path}")
            continue
        
        try:
            with fitz.open(stream=pdf_path.read_bytes(), filetype="pdf") as doc:
                pages = [page.get_text("text") for page in doc]
            text = "\n".join(pages).strip()
        except Exception as e:
            # PDF処理失敗時、テキストとして処理を試みる
            try:
                text = pdf_path.read_bytes().decode('utf-8', errors='ignore')
                if not text.strip():
                    print(f"  [WARN] Could not extract text from {pdf_path}: {type(e).__name__}")
                    continue
            except Exception:
                print(f"  [WARN] Could not process {pdf_path}: {type(e).__name__}: {e}")
                continue
        
        # テキストを同じディレクトリ構造で保存
        rel_pdf = pdf_path.relative_to(base_root / "pdf_files")
        text_path = text_dir / rel_pdf.with_suffix(".txt")
        save_pdf_text(text, text_path)
        
        text_rows.append({
            "code": row["code"],
            "company_name": row["company_name"],
            "title": row["title"],
            "pubdate": row["pubdate"],
            "text_path": str(text_path),
        })
    
    if text_rows:
        # メタデータを pdf_metadata フォルダに保存
        meta_dir = meta_path.parent
        text_meta_path = meta_dir / meta_path.name.replace("pdf_meta_", "pdf_text_meta_")
        text_meta_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(text_rows).to_csv(text_meta_path, index=False, encoding="utf-8")
        print(f"  saved text metadata: {text_meta_path}")


def extract_words_from_saved_text(text_meta_csv: str | Path, base_root: str | Path | None = None) -> None:
    """
    抽出済みテキストファイルから単語を抽出して CSV として保存する。
    base_root が None の場合、text_meta_csv と同じ csv フォルダを基準とします。
    """
    text_meta_path = Path(text_meta_csv)
    if base_root is None:
        base_root = text_meta_path.parent.parent.parent
    base_root = Path(base_root)
    
    df_text = pd.read_csv(text_meta_csv)
    word_dir = base_root / "pdf_words"
    word_rows = []
    
    print(f"extract_words_from_saved_text: processing {len(df_text)} texts...")
    for idx, row in df_text.iterrows():
        text_path = Path(row["text_path"])
        if not text_path.exists():
            print(f"  [WARN] Text file not found: {text_path}")
            continue
        
        try:
            text = text_path.read_text(encoding="utf-8")
            words = extract_pdf_words(text)
        except Exception as e:
            print(f"  [WARN] Could not extract words from {text_path}: {type(e).__name__}: {e}")
            continue
        
        # 単語CSVを保存
        word_path = word_dir / text_path.relative_to(base_root / "pdf_text").with_suffix("_words.csv")
        save_pdf_words(words, word_path)
        
        word_rows.append({
            "code": row["code"],
            "company_name": row["company_name"],
            "title": row["title"],
            "pubdate": row["pubdate"],
            "text_path": row["text_path"],
            "word_path": str(word_path),
            "word_count": len(words),
        })
    
    if word_rows:
        # メタデータを保存
        meta_dir = text_meta_path.parent
        word_meta_path = meta_dir / text_meta_path.name.replace("pdf_text_meta_", "pdf_word_meta_")
        word_meta_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(word_rows).to_csv(word_meta_path, index=False, encoding="utf-8")
        print(f"  saved word metadata: {word_meta_path}")




def sanitize_filename(text: str, max_length: int = 120) -> str:
    invalid_chars = '<>:"/\\|?*\n\r\t'
    cleaned = "".join("_" if ch in invalid_chars else ch for ch in text)
    return cleaned[:max_length].strip(" _") or "unnamed"


def extract_html_from_zip(zip_data: bytes, log_root: Path | None = None):
    """
    ZIPアーカイブからHTMLまたはiXBRLを抽出
    （PDFが無い場合のフォールバック）
    """
    try:
        with zipfile.ZipFile(io.BytesIO(zip_data)) as z:
            all_files = z.namelist()
            
            # 優先順: qualitative.htm > ixbrl.htm > その他.htm
            preferred_files = [
                f for f in all_files 
                if 'qualitative.htm' in f.lower()
            ]
            if not preferred_files:
                preferred_files = [
                    f for f in all_files 
                    if f.lower().endswith('.htm') or f.lower().endswith('.html')
                ]
            
            if not preferred_files:
                msg = f"[HTML EXTRACT] No HTML files found in ZIP"
                print(msg)
                if log_root is not None:
                    log_error(msg, log_root)
                return None
            
            # 最初のHTMLファイルを取得
            html_file = preferred_files[0]
            msg = f"[HTML EXTRACT] Extracting from: {html_file}"
            print(msg)
            if log_root is not None:
                log_error(msg, log_root)
            
            html_data = z.read(html_file).decode('utf-8', errors='ignore')
            
            # HTMLからテキストを抽出
            class HTMLTextExtractor(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.text_parts = []
                    self.skip_script = False
                
                def handle_starttag(self, tag, attrs):
                    if tag in ('script', 'style'):
                        self.skip_script = True
                
                def handle_endtag(self, tag):
                    if tag in ('script', 'style'):
                        self.skip_script = False
                
                def handle_data(self, data):
                    if not self.skip_script:
                        text = data.strip()
                        if text:
                            self.text_parts.append(text)
                
                def get_text(self):
                    return '\n'.join(self.text_parts)
            
            extractor = HTMLTextExtractor()
            extractor.feed(html_data)
            text = extractor.get_text()
            
            if text.strip():
                msg = f"[HTML EXTRACT] Extracted {len(text)} characters from HTML"
                print(msg)
                if log_root is not None:
                    log_error(msg, log_root)
                
                # テキストをバイト列に変換して返す
                class DownloadResult:
                    def __init__(self, data):
                        self.data = data
                
                return DownloadResult(text.encode('utf-8'))
            else:
                msg = f"[HTML EXTRACT] HTML file is empty or unreadable"
                print(msg)
                if log_root is not None:
                    log_error(msg, log_root)
                return None
    
    except Exception as e:
        msg = f"[HTML EXTRACT ERROR] {type(e).__name__}: {e}"
        print(msg)
        if log_root is not None:
            log_error(msg, log_root)
        return None


def extract_pdf_from_zip(zip_data: bytes, log_root: Path | None = None):
    """
    ZIPアーカイブからPDFを抽出
    （サブディレクトリ内のPDFも検索）
    """
    try:
        with zipfile.ZipFile(io.BytesIO(zip_data)) as z:
            # ZIP内のすべてのファイルをログ出力
            all_files = z.namelist()
            msg = f"[ZIP EXTRACT] ZIP contents: {all_files}"
            print(msg)
            if log_root is not None:
                log_error(msg, log_root)
            
            # 再帰的にPDFを検索
            pdf_files = [f for f in all_files if f.lower().endswith('.pdf')]
            
            if not pdf_files:
                msg = f"[ZIP EXTRACT] No PDF found in ZIP, trying HTML fallback..."
                print(msg)
                if log_root is not None:
                    log_error(msg, log_root)
                # HTML抽出へフォールバック
                return extract_html_from_zip(zip_data, log_root=log_root)
            
            # 最初のPDFを取得
            pdf_file = pdf_files[0]
            msg = f"[ZIP EXTRACT] Found PDF in ZIP: {pdf_file}"
            print(msg)
            if log_root is not None:
                log_error(msg, log_root)
            
            pdf_data = z.read(pdf_file)
            
            # DownloadResult互換オブジェクトを返す
            class DownloadResult:
                def __init__(self, data):
                    self.data = data
            
            return DownloadResult(pdf_data)
    
    except Exception as e:
        msg = f"[ZIP EXTRACT ERROR] {type(e).__name__}: {e}"
        print(msg)
        if log_root is not None:
            log_error(msg, log_root)
        return None


def fetch_pdf_with_fallback(filing, log_root: Path | None = None):
    """
    fetch_pdf失敗時に、requestsで直接URLから取得するフォールバック処理
    （ZIPアーカイブ対応）
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    # 試行するURL一覧
    urls_to_try = []
    if filing.document_url:
        urls_to_try.append(("document_url", filing.document_url))
    if hasattr(filing, "xbrl_url") and filing.xbrl_url:
        urls_to_try.append(("xbrl_url", filing.xbrl_url))
    
    for url_type, url in urls_to_try:
        try:
            response = requests.get(
                url,
                headers=headers,
                allow_redirects=True,
                timeout=30,
                verify=False
            )
            response.raise_for_status()
            
            content_type = response.headers.get("Content-Type", "")
            
            # PDFデータか確認
            if content_type.startswith("application/pdf"):
                class DownloadResult:
                    def __init__(self, data):
                        self.data = data
                
                return DownloadResult(response.content)
            
            # ZIPファイルか確認
            elif content_type.startswith("application/zip"):
                result = extract_pdf_from_zip(response.content, log_root=log_root)
                if result is not None:
                    return result
        
        except Exception:
            # エラーは無視してスキップ（古いPDFなど確実に失敗するケース対応）
            pass
    
    return None


def extract_pdf_text(filing, log_root: Path | None = None) -> str | None:
    result = None
    
    # 第1段階：tdnet.fetch_pdf を試行
    try:
        result = retry_on_404(
            filing.fetch_pdf,
            retries=1,
            wait_seconds=1.0,
            log_root=log_root,
            description=f"filing.fetch_pdf for {filing.company_code} {filing.title}",
        )
    except Exception:
        # 第2段階：requestsでのフォールバック処理
        result = fetch_pdf_with_fallback(filing, log_root=log_root)
    
    if result is None:
        # ログは記録しない（古いPDFは確実に失敗するため）
        return None

    try:
        # PDFとして処理
        with fitz.open(stream=result.data, filetype="pdf") as doc:
            pages = [page.get_text("text") for page in doc]
        return "\n".join(pages).strip()
    except Exception as e:
        # PDF処理失敗時、テキストとして処理を試みる
        msg = f"[PDF PARSE ERROR] {filing.company_code} {filing.title}: Trying as text... ({type(e).__name__})"
        print(msg)
        if log_root is not None:
            log_error(msg, log_root)
        
        try:
            text = result.data.decode('utf-8', errors='ignore')
            if text.strip():
                return text.strip()
        except Exception:
            pass
        
        msg = f"[PDF PARSE FAILED] {filing.company_code} {filing.title}: {type(e).__name__}: {e}"
        print(msg)
        if log_root is not None:
            log_error(msg, log_root)
        return None


def extract_pdf_words(text: str) -> list[str]:
    tokenizer = Tokenizer()
    words: list[str] = []
    for token in tokenizer.tokenize(text):
        pos = token.part_of_speech.split(",", 1)[0]
        if pos in {"名詞", "動詞", "形容詞"}:
            base = token.base_form
            word = base if base != "*" and base != token.surface else token.surface
            if word.strip():
                words.append(word)
    return words


def save_pdf_text(text: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def save_pdf_words(words: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"word": words}).to_csv(output_path, index=False, encoding="utf-8")


def save_pdf_binary(pdf_data: bytes, output_path: Path) -> None:
    """
    PDFファイルをバイナリとして保存する
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(pdf_data)


def process_pdf_files(
    filings,
    year: int,
    date_str: str,
    base_root: Path,
    log_root: Path | None = None,
) -> None:
    """
    PDFをファイルとして保存し、メタデータを記録する。
    テキスト・単語抽出は別途 extract_text_from_saved_pdfs() で実施する。
    """
    pdf_root = base_root / "pdf_files" / str(year) / date_str
    meta_rows = []

    print(f"process_pdf_files {len(filings)} filings...")
    for filing in filings:
        filename_base = f"{filing.company_code}_{sanitize_filename(filing.title or filing.document_url or filing.xbrl_url)}"
        try:
            pdf_result = retry_on_404(
                filing.fetch_pdf,
                retries=1,
                wait_seconds=1.0,
                log_root=log_root,
                description=f"filing.fetch_pdf for {filing.company_code} {filing.title}",
            )
        except Exception:
            # フォールバック処理を試行
            pdf_result = fetch_pdf_with_fallback(filing, log_root=log_root)
        
        if pdf_result is None:
            continue

        pdf_path = pdf_root / f"{filename_base}.pdf"
        save_pdf_binary(pdf_result.data, pdf_path)

        meta_rows.append(
            {
                "code": filing.company_code,
                "company_name": filing.company_name,
                "title": filing.title,
                "pubdate": filing.pubdate,
                "pdf_path": str(pdf_path),
            }
        )

    if meta_rows:
        meta_path = base_root / "pdf_metadata" / str(year) / f"pdf_meta_{date_str}.csv"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_df = pd.DataFrame(meta_rows)
        meta_df.to_csv(meta_path, index=False, encoding="utf-8")

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
# 5. 実行例
# メイン実行部: 年を指定して処理を開始する
# ----------------------------------------
if __name__ == "__main__":
    #year = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
    year = 2023
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