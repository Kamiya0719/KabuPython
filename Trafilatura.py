import trafilatura
import pandas as pd
import json
from Labeling import classify_news
import csv
import os
import time
import re
import unicodedata
from bs4 import BeautifulSoup



# Trafilatura を使って記事の URL からタイトルとテキストを抽出する関数
def extract_article_with_metadata(url):
    downloaded = trafilatura.fetch_url(url)
    if downloaded is None:
        return None
    data = trafilatura.extract(downloaded, output_format="json", with_metadata=True)
    if data is None:
        return None
    return json.loads(data)


# Title と Text から openaiにてニュースのラベルつけを行う関数
def SetLabelReason(input_csv, output_csv, num=100):
    with open(input_csv, "r", newline="", encoding="utf-8") as fin, \
        open(output_csv, "w", newline="", encoding="utf-8") as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        start = time.time()
        cnt = 0
        for i, row in enumerate(reader):
            # Ensure the row has at least 11 columns (indices 0..10)
            while len(row) < 11:
                row.append("")
            
            labelNow = row[9]
            # todo タイムアウトのやつはどうしようか？
            if i > 0 and cnt <= num and (labelNow is None or labelNow == ""):
                title = row[7]
                text = row[8]
                if title is not None and text is not None and len(text) > 10:
                    print("Processing:", cnt, i, time.time() - start)
                    result = classify_news(title, text)
                    row[9] = result["label"]
                    row[10] = result["reason"]
                else:
                    row[9] = "NODATA"
                    row[10] = "NODATA"
                cnt += 1                    

            writer.writerow(row)

    os.replace(output_csv, input_csv)


# GDELTデータのURLからニュースのタイトルとテキストを抽出して保存する関数
def SetTitleText(input_csv, output_csv):
    with open(input_csv, "r", newline="", encoding="utf-8") as fin, \
        open(output_csv, "w", newline="", encoding="utf-8") as fout:

        reader = csv.reader(fin)
        writer = csv.writer(fout)
        start = time.time()

        for i, row in enumerate(reader):
            # Ensure the row has at least 11 columns (indices 0..10)
            while len(row) < 11:
                row.append("")

            # 1行目はヘッダーなのでスキップ
            if i != 0 :
                url = row[2]
                print("Processing:", i, time.time() - start)
                data = extract_article_with_metadata(url)
                if data is not None:
                    title = data.get("title") or ""
                    title = TextReplace(title)
                    text = data.get("text") or ""
                    text = TextReplace(text)
                    if len(title) > 100:
                        title = title[:100]
                    if len(text) > 500:
                        text = text[:500]
                    row[7] = title
                    row[8] = text
                else:
                    row[7] = "NODATA"
                    row[8] = "NODATA"

            writer.writerow(row)

    #os.replace(output_csv, input_csv)


# デバッグ用あれこれ
def ExecDebug(input_csv, output_csv):
    with open(input_csv, "r", newline="", encoding="utf-8") as fin, \
        open(output_csv, "w", newline="", encoding="utf-8") as fout:

        reader = csv.reader(fin)
        writer = csv.writer(fout)
        start = time.time()

        for i, row in enumerate(reader):
            # Ensure the row has at least 11 columns (indices 0..10)
            while len(row) < 11:
                row.append("")

            # 1行目はヘッダーなのでスキップ
            if i != 0 :
                row[7] = TextReplace(row[7])
                row[7] = row[7].replace(",", " ")
                row[8] = TextReplace(row[8])
                row[8] = row[8].replace(",", " ")
                if row[9] == "NoData":
                    row[9] = "NODATA"
                if row[9] == "1. Micro":
                    row[9] = "Micro"
                if row[9] not in LABELS:
                    row[9] = ""
            writer.writerow(row)

    #os.replace(output_csv, input_csv)


# テキストをクリーンアップする関数
def TextReplace(text: str) -> str:
    # タイトルと本文を結合
    raw = text
    # HTMLタグ除去
    raw = BeautifulSoup(raw, "html.parser").get_text()
    # 改行 → スペース
    raw = raw.replace("\n", " ").replace("\r", " ")
    # 全角半角正規化
    raw = unicodedata.normalize("NFKC", raw)
    # 不要な記号の簡易除去（株価予測ではノイズになりやすい）
    raw = re.sub(r"[■◆▲▽▶◀★☆●○◎◇◆※]", " ", raw)
    # 連続スペースを 1 個に圧縮
    raw = re.sub(r"\s+", " ", raw).strip()
    return raw


LABELS = [
    "Micro",
    "Sector Macro",
    "Country Macro",
    "Global Macro",
    "Mixed",
    "Other",
    "NODATA"
]
def CsvDispDebug(input_csv, num=10):
    with open(input_csv, "r", newline="", encoding="utf-8") as fin:
        reader = csv.reader(fin)
        cnt = 0
        labelInfos = {}
        for i, row in enumerate(reader):
            if i == 0:
                continue
            label = row[9]
            #if label is not None and label != "":
            if label != "" and label not in LABELS:
                print(f"Unexpected label at line {i}: '{label}'")
                continue

            labelInfos[label] = labelInfos.get(label, 0) + 1
        for label, count in labelInfos.items():
            print(f"Label: {label}, Count: {count}")



def CheckMicroOrgan(input_csv, num=10):
    with open(input_csv, "r", newline="", encoding="utf-8") as fin:
        reader = csv.reader(fin)
        cnt = 0
        labelInfos = {}
        for i, row in enumerate(reader):
            if i == 0:
                continue
            label = row[9]
            if label != "Micro":
                continue
            themes = row[4]
            organizations = row[5]

            labelInfos[label] = labelInfos.get(label, 0) + 1
        for label, count in labelInfos.items():
            print(f"Label: {label}, Count: {count}")




"""
    # 3列目(2), 5列目(4), 6列目(5) を取得
    subset = df.iloc[start:end, [2, 4, 5]]

    results = []
    for idx, row in subset.iterrows():
        url = row[2]
        V2Themes = row[4]
        V2Organizations = row[5]

        print("Processing:", url)
        data = extract_article_with_metadata(url)

        if data is not None:

            result = classify_news(data.get("title"), data.get("text"), V2Organizations, V2Themes)
            label = result["label"]
            reason = result["reason"]


            results.append({
                "idx": idx,
                "label": label,
                "reason": reason,
                "V2Themes": V2Themes,
                "V2Organizations": V2Organizations
            })

    # 保存
    out_df = pd.DataFrame(results)
    out_df.to_csv(output_csv, index=False)
    #return results

"""


for i in range(0, 15):
    # 特定日付のurl・テーマ・組織　のデータを処理して、ラベルと理由をCSVに保存する
    inputPath = r"C:\Users\ojiro\Documents\PythonFolder\KabuPython\20260525.csv"
    outputPath = r"C:\Users\ojiro\Documents\PythonFolder\KabuPython\20260525Hoge.csv"
    SetLabelReason(inputPath, outputPath, 300)



# 特定日付のurl・テーマ・組織　のデータを処理して、ラベルと理由をCSVに保存する
inputPath = r"C:\Users\ojiro\Documents\PythonFolder\KabuPython\20260525.csv"
outputPath = r"C:\Users\ojiro\Documents\PythonFolder\KabuPython\20260525Hoge.csv"
#SetLabelReason(inputPath, outputPath)
CsvDispDebug(inputPath)

