import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import glob
import argparse
from pathlib import Path
import numpy as np

# === 設定 ===
TECHNICAL_SCORE_GLOB = "C:/Users/ojiro/Documents/KabuCSharp/KabuCSharp/KabuCSharp/csv/Debug/TechnicalScore/*.csv"
MODEL_BASE = "C:/Users/ojiro/Documents/PythonFolder/KabuPython"

ALL_FEATURES = [
    "dateIdx", "HistoricalVorality", "RollingStd20", "RollingStd60", "ATR", "BollingerBand", "RangeRatio", "VolRollingStd", "VolumeROC",
    "BullishRate","AverageReturn", "AverageRSI", "AverageMACDHistogram", "TrendSlope", "VolumeTrendStrength", "DemandSupplyScore", "BreakoutReliability",
    "BigDown", "GapDown", "LowerWick", "PanicSell", "VolumeSpikeScore", "VolumeBasedRisk",
    "SalesYoY", "OperatingMargin", "EpsYoY", "ROE", "CFORatio", "EqAR", "PER", "PBR", "MarketCap", "EPS", "CFYYield", "CFO",
    "MarginBalanceRatio", "IssType", "MarginLongRatio", "MarginShortRatio", "ShrtPosToSO", "ShortSaleChange", "ShrtPosShares",
    "ShortMarginChange", "LongMarginChange", "OutChgRatio", "SLRatio",
    "close",
    "ROC5", "ROC10", "ROC20",
    "EMA5", "EMA20", "EMA60",
    "EMA5_div", "EMA20_div", "EMA60_div",
    "MACD_diff", "RSI_delta", "open","high", "low", "yearLowest", "yearHighest", "monthLowest", "monthHighest",
    "secondMonthHigh", "secondMonthLow", "bunsanLowM", "bunsanHighM", "bunsanLowW", "bunsanHighW",
    "bunsanLowD", "bunsanHighD", "lowHosyou", "highHosyou",
]

# 最終予測のpredicted_benefitのために追加する特徴量
PRED_FEATURES = [
    "predicted_futureStability","predicted_futureUp","predicted_futureFall",
    "predicted_nextHigh","predicted_nextLow",
]

# ============================================================
# saveModel
# ============================================================
def saveModel(type, prediction_dir=None):
    files = glob.glob(TECHNICAL_SCORE_GLOB)
    if not files:
        raise FileNotFoundError("学習対象CSVがありません")

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df["symbol"] = Path(f).stem
        df = add_technical_features(df, isAllDate=True)
        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)
    if type == 2:
        all_df = attach_prediction_features(all_df, files, prediction_dir)
    all_df = all_df.replace(-99, pd.NA)

    targets = ["futureStability", "futureUp", "futureFall","nextHigh", "nextLow"]
    if type == 2:
        targets = ["futureBenefit"]


    for t in targets:
        base_features = ALL_FEATURES.copy()
        if t == "futureBenefit":
            base_features += PRED_FEATURES
        is_classification = t == "futureBenefit"
        selected = auto_feature_selection(
            all_df, t, base_features, threshold_ratio=0.01,
            is_classification=is_classification
        )

        # 追加：特徴量リストを保存
        with open(f"{MODEL_BASE}/{t}_features.txt", "w") as f:
            for feat in selected:
                f.write(feat + "\n")

        # 自動選択された特徴量で再学習
        X = all_df[selected].apply(pd.to_numeric, errors="coerce")
        y = all_df[t].apply(pd.to_numeric, errors="coerce")
        if is_classification:
            y = y.astype("category").cat.codes.astype("float32")
            valid_labels = y.notna() & y.ge(0)
            X = X.loc[valid_labels]
            y = y.loc[valid_labels].astype("int32")

        # --- 追加：目的変数の分布を表示 ---
        y_desc = y.describe()
        print(f"[{t}] Target Distribution")
        print(f" min: {y_desc['min']}, max: {y_desc['max']}, mean: {y_desc['mean']}, std: {y_desc['std']}")

        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_valid, label=y_valid)

        params = {
            "objective": "multiclass" if is_classification else "regression",
            "metric": "multi_logloss" if is_classification else "rmse",
            "learning_rate": 0.05,
            "num_leaves": 64,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
        }
        if is_classification:
            params["num_class"] = int(y.max()) + 1

        model = lgb.train(
            params,
            train_data,
            num_boost_round=2000,
            valid_sets=[train_data, valid_data],
            valid_names=["train", "valid"],
            callbacks=[lgb.early_stopping(stopping_rounds=100)]
        )

        if is_classification:
            train_loss = model.best_score["train"]["multi_logloss"]
            valid_loss = model.best_score["valid"]["multi_logloss"]
            print(f"[{t}] train logloss: {train_loss:.6f}, valid logloss: {valid_loss:.6f}, best iteration: {model.best_iteration}")
        else:
            train_rmse = model.best_score["train"]["rmse"]
            valid_rmse = model.best_score["valid"]["rmse"]

            # --- 追加：RMSE の評価 ---
            rmse_ratio = valid_rmse / y_desc["std"]
            print(f"[{t}] train RMSE: {train_rmse:.6f}, valid RMSE: {valid_rmse:.6f}, best iteration: {model.best_iteration}")
            print(f"[{t}] RMSE/std: {rmse_ratio:.4f}  →  評価: ", end="")
            if rmse_ratio < 0.3:
                print("非常に良い（高い汎化性能）")
            elif rmse_ratio < 0.5:
                print("良い（実用レベル）")
            elif rmse_ratio < 0.8:
                print("普通（改善余地あり）")
            else:
                print("弱い（特徴量の見直し推奨）")



        import numpy as np
        # --- 予測 ---
        pred = model.predict(X_valid)
        if is_classification:
            pred = pred.argmax(axis=1)
        y_values = pd.to_numeric(y_valid, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
        eval_mask = np.isfinite(y_values) & np.isfinite(pred)
        y_eval = y_values[eval_mask]
        pred_eval = pred[eval_mask]

        if len(y_eval) == 0:
            print(f"[{t}] Hit Rate: N/A (評価可能なデータがありません)")
            model.save_model(f"{MODEL_BASE}/{t}model.txt")
            continue

        # --- 上位25%・下位25% の閾値 ---
        top_q = 0.75
        bottom_q = 0.25
        if t == "futureBenefit":
            top_q = 0.97
            bottom_q = 0.86
        y_top_thr = np.quantile(y_eval, top_q)
        y_bottom_thr = np.quantile(y_eval, bottom_q)
        pred_top_thr = np.quantile(pred_eval, top_q)
        pred_bottom_thr = np.quantile(pred_eval, bottom_q)
        # --- インデックス抽出 ---
        y_top_idx = set(np.where(y_eval >= y_top_thr)[0])
        y_bottom_idx = set(np.where(y_eval <= y_bottom_thr)[0])
        pred_top_idx = set(np.where(pred_eval >= pred_top_thr)[0])
        pred_bottom_idx = set(np.where(pred_eval <= pred_bottom_thr)[0])
        # --- 一致率（Hit Rate） ---
        top_hit_rate = len(y_top_idx & pred_top_idx) / len(y_top_idx) if y_top_idx else float("nan")
        bottom_hit_rate = len(y_bottom_idx & pred_bottom_idx) / len(y_bottom_idx) if y_bottom_idx else float("nan")
        print(f"[{t}] Top 20% Hit Rate:    {top_hit_rate:.4f}")
        print(f"[{t}] Bottom 20% Hit Rate: {bottom_hit_rate:.4f}")

        #特徴量重要度を表示するコード（追加推奨）
        importance = model.feature_importance()
        feature_names = model.feature_name()
        disp = f"[{t}] Feature Importance"
        for name, imp in sorted(zip(feature_names, importance), key=lambda x: -x[1]):
            disp += f"  {name}: {imp}"
        print(disp + f"\n\n")


        model.save_model(f"{MODEL_BASE}/{t}model.txt")


def attach_prediction_features(all_df, technical_files, prediction_dir=None):
    """予測出力CSVから type==2 用の予測特徴量を元データへ結合する。"""
    if prediction_dir is None:
        prediction_dir = Path(technical_files[0]).parent / "daily_predicted"
    else:
        prediction_dir = Path(prediction_dir)

    prediction_files = [
        prediction_dir / f"{Path(technical_file).stem}_pred.csv"
        for technical_file in technical_files
    ]
    existing_files = [path for path in prediction_files if path.exists()]
    if not existing_files:
        raise FileNotFoundError(f"予測特徴量CSVがありません: {prediction_dir}")

    prediction_frames = []
    for prediction_file in existing_files:
        prediction_df = pd.read_csv(prediction_file)
        required_columns = {"symbol", "dateIdx", *PRED_FEATURES}
        missing_columns = required_columns - set(prediction_df.columns)
        if missing_columns:
            raise ValueError(
                f"予測特徴量CSVに必要な列がありません ({prediction_file}): "
                f"{sorted(missing_columns)}"
            )
        prediction_frames.append(prediction_df[["symbol", "dateIdx", *PRED_FEATURES]])

    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions = predictions.drop_duplicates(["symbol", "dateIdx"], keep="last")
    return all_df.merge(predictions, on=["symbol", "dateIdx"], how="left")

# ============================================================
# 特徴量選択
# ============================================================
def auto_feature_selection(all_df, target_name, base_features, threshold_ratio=0.01, is_classification=False):
    X = all_df[base_features].apply(pd.to_numeric, errors="coerce")
    y = all_df[target_name].apply(pd.to_numeric, errors="coerce")
    if is_classification:
        y = y.astype("category").cat.codes.astype("float32")
        valid_labels = y.notna() & y.ge(0)
        X = X.loc[valid_labels]
        y = y.loc[valid_labels].astype("int32")

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = lgb.train(
        {
            "objective": "multiclass" if is_classification else "regression",
            "metric": "multi_logloss" if is_classification else "rmse",
            "learning_rate": 0.05,
            "num_leaves": 64,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            **({"num_class": int(y.max()) + 1} if is_classification else {})
        },
        lgb.Dataset(X_train, label=y_train),
        num_boost_round=500,
        valid_sets=[lgb.Dataset(X_valid, label=y_valid)],
        valid_names=["valid"],
        callbacks=[lgb.early_stopping(stopping_rounds=50)]
    )

    selected = select_features_by_importance(model, model.feature_name(), threshold_ratio)
    print(f"Selected Features ({len(selected)}): {selected}")
    return selected

def select_features_by_importance(model, feature_names, threshold_ratio=0.01):
    importance = model.feature_importance()
    total_importance = sum(importance)
    return [name for name, imp in zip(feature_names, importance) if imp / total_importance >= threshold_ratio]





# ============================================================
# 全銘柄 predict（高速版）
# ============================================================
def predict_all(type, output_dir=None, model_dir=".", date_idx=None):
    files = glob.glob(TECHNICAL_SCORE_GLOB)
    if not files:
        raise FileNotFoundError("予測対象CSVがありません")

    if output_dir is None:
        output_dir = Path(files[0]).parent / "daily_predicted"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models, features = load_models(type)

    output_paths = []
    for input_csv in files:
        output_csv = output_dir / (Path(input_csv).stem + "_pred.csv")
        result = predict_and_save(type, input_csv, output_csv, models, features, date_idx)
        if result is not None:
            output_paths.append(result)
    return output_paths

# ============================================================
# モデルロード（高速化）
# ============================================================
def load_models(type):
    targets = ["futureStability", "futureUp", "futureFall", "nextHigh", "nextLow"]
    if type == 2:
        targets.append("futureBenefit")
    models = {}
    features = {}
    for t in targets:
        models[t] = lgb.Booster(model_file=str(Path(MODEL_BASE) / f"{t}model.txt"))
        with open(Path(MODEL_BASE) / f"{t}_features.txt") as f:
            features[t] = [line.strip() for line in f.readlines()]

    return models, features

# ============================================================
# predict（複数行保存）
# ============================================================
def predict_and_save(type, input_csv, output_csv, models, features, date_idx=None):
    df = pd.read_csv(input_csv) # technicalScore CSV を読み込む
    isAllDate = date_idx is None
    df = add_technical_features(df, isAllDate=isAllDate) # テクニカル指標を追加

    if isAllDate: # dateIdx が指定されていない場合は全行を対象
        target_df = df
    else:
        target_df = df[df["dateIdx"].isin(date_idx)]
    if target_df.empty:
        return None

    results = []
    targets = ["futureStability", "futureUp", "futureFall", "nextHigh", "nextLow"]
    if type == 2:
        targets.append("futureBenefit")

    previous_predictions = None
    if type == 2 and Path(output_csv).exists():
        previous_predictions = pd.read_csv(output_csv).set_index("dateIdx")

    # 各日付で処理を行う
    for _, row in target_df.iterrows():
        row_values = row.to_dict()
        if previous_predictions is not None and row["dateIdx"] in previous_predictions.index:
            previous_row = previous_predictions.loc[row["dateIdx"]]
            for feature in PRED_FEATURES:
                if feature in previous_row and pd.notna(previous_row[feature]):
                    row_values[feature] = previous_row[feature]
        row_result = {
            "symbol": row["symbol"] if "symbol" in df.columns else Path(input_csv).stem,
            "dateIdx": row["dateIdx"],
        }

        for t in targets:
            model = models[t]
            selected_features = features[t]

            prediction_key = f"predicted_{t}"
            if type == 2 and t != "futureBenefit" and pd.notna(row_values.get(prediction_key)):
                predicted_value = float(row_values[prediction_key])
                if predicted_value != -99:
                    row_result[prediction_key] = predicted_value
                    continue

            missing_features = [
                feature for feature in selected_features if feature not in row_values
            ]
            if missing_features:
                raise ValueError(
                    f"{t} の予測に必要な特徴量がありません: {missing_features}"
                )
            feature_data = pd.DataFrame(
                [{feature: row_values[feature] for feature in selected_features}
            ]).astype(float)
            pred = model.predict(feature_data)

            if type == 2 and t == "futureBenefit":
                pred = pred.argmax(axis=1) + 1 # 1から始まるクラスラベルに変換

            predicted_value = float(pred[0])
            row_values[prediction_key] = predicted_value
            row_result[prediction_key] = predicted_value

        results.append(row_result)

    # === ここから append 保存 ===
    new_df = pd.DataFrame(results)

    if Path(output_csv).exists():
        # 既存ファイルを読み込み
        old_df = pd.read_csv(output_csv)

        # 重複日付を避ける（同じ dateIdx があれば上書き）
        merged = pd.concat([old_df[~old_df["dateIdx"].isin(new_df["dateIdx"])], new_df], ignore_index=True)
        merged = merged.sort_values("dateIdx")
        merged.to_csv(output_csv, index=False)
    else:
        # 初回は普通に保存
        new_df.to_csv(output_csv, index=False)

    return output_csv



# ============================================================
# テクニカル指標（直近70日）
# ============================================================
def add_technical_features(df, isAllDate=False):
    if isAllDate:
        df = df.copy()
    else:
        df = df.tail(70).copy()

    df["ROC5"] = df["close"].pct_change(5)
    df["ROC10"] = df["close"].pct_change(10)
    df["ROC20"] = df["close"].pct_change(20)

    df["EMA5"] = df["close"].ewm(span=5).mean()
    df["EMA20"] = df["close"].ewm(span=20).mean()
    df["EMA60"] = df["close"].ewm(span=60).mean()

    df["EMA5_div"] = (df["close"] - df["EMA5"]) / df["EMA5"]
    df["EMA20_div"] = (df["close"] - df["EMA20"]) / df["EMA20"]
    df["EMA60_div"] = (df["close"] - df["EMA60"]) / df["EMA60"]

    df["EMA12"] = df["close"].ewm(span=12).mean()
    df["EMA26"] = df["close"].ewm(span=26).mean()
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["MACD_signal"] = df["MACD"].ewm(span=9).mean()
    df["MACD_diff"] = df["MACD"] - df["MACD_signal"]

    diff = df["close"].diff()
    gain = diff.clip(lower=0)
    loss = -diff.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    df["RSI_delta"] = df["RSI"].diff()

    return df



# ============================================================
# main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="高速化版テクニカル予測")
    parser.add_argument("action", choices=["save", "predict"])
    parser.add_argument("type", type=int)
    parser.add_argument("--date-idx", nargs="*", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--prediction-dir", default=None)
    parser.add_argument("--model-dir", default=".")
    args = parser.parse_args()

    if args.action == "save":
        saveModel(args.type, args.prediction_dir)
    else:
        predict_all(args.type, args.output_dir, args.model_dir, args.date_idx)


if __name__ == "__main__":
    main()
