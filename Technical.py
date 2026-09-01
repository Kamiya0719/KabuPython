import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import glob
import argparse
from pathlib import Path


# テクニカル分析の指標から未来値予測を行う学習モデルを生成。

ALL_FEATURES = [
    "dateIdx", "HistoricalVorality", "RollingStd20", "RollingStd60", "ATR", "BollingerBand", "RangeRatio", "VolRollingStd", "VolumeROC",
    # テクニカル（上昇系）
    "BullishRate","AverageReturn", "AverageRSI", "AverageMACDHistogram", "TrendSlope", "VolumeTrendStrength", "DemandSupplyScore", "BreakoutReliability",
    # テクニカル（下落系）
    "BigDown", "GapDown", "LowerWick", "PanicSell", "VolumeSpikeScore", "VolumeBasedRisk",
    # 財務系
    "SalesYoY", "OperatingMargin", "EpsYoY", "ROE", "CFORatio", "EqAR", "PER", "PBR", "MarketCap", "EPS", "CFYYield", "CFO",
    # 信用・需給系
    "MarginBalanceRatio", "IssType", "MarginLongRatio", "MarginShortRatio", "ShrtPosToSO", "ShortSaleChange", "ShrtPosShares",
    "ShortMarginChange", "LongMarginChange", "OutChgRatio", "SLRatio",
    "close",
    "ROC5", "ROC10", "ROC20",
    "EMA5", "EMA20", "EMA60",
    "EMA5_div", "EMA20_div", "EMA60_div",
    "MACD_diff", "RSI_delta", "open","high", "low", "yearLowest", "yearHighest", "monthLowest", "monthHighest", "secondMonthHigh", "secondMonthLow", "bunsanLowM",
 "bunsanHighM", "bunsanLowW", "bunsanHighW", "bunsanLowD", "bunsanHighD", "lowHosyou", "highHosyou", 
]
PRED_FEATURES = [
    "predicted_futureStability","predicted_futureUp","predicted_futureFall","predicted_nextHigh",
    "predicted_nextLow",
]

TECHNICAL_SCORE_GLOB = "C:/Users/ojiro/Documents/KabuCSharp/KabuCSharp/KabuCSharp/csv/Debug/TechnicalScore/*.csv"
MODEL_BASE = "C:/Users/ojiro/Documents/PythonFolder/KabuPython"

def saveModel(type):
    # 1. CSV を読み込む

    files = glob.glob(TECHNICAL_SCORE_GLOB)

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df["symbol"] = f.split("\\")[-1].replace(".csv", "")
        df = add_technical_features(df)
        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)
    all_df = all_df.replace(-99, pd.NA)

    targets = ["futureStability", "futureUp", "futureFall","nextHigh", "nextLow"]
    if type == 2:
        #targets = ["futureBenefit","futureBenefitDate"]
        targets = ["futureBenefit"]


    for t in targets:
        base_features = ALL_FEATURES.copy()
        if type == 2:
            base_features += PRED_FEATURES
        is_classification = type == 2 and t == "futureBenefit"
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
        #X = all_df[features].apply(pd.to_numeric, errors="coerce")
        #y = all_df[t].apply(pd.to_numeric, errors="coerce")

        # --- 追加：目的変数の分布を表示 ---
        y_desc = y.describe()
        print(f"[{t}] Target Distribution")
        print(f" min:  {y_desc['min']:.6f}, max:  {y_desc['max']:.6f}, mean: {y_desc['mean']:.6f}, std:  {y_desc['std']:.6f}")

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
        callbacks = [lgb.early_stopping(stopping_rounds=100)]

        model = lgb.train(
            params,
            train_data,
            num_boost_round=2000,
            valid_sets=[train_data, valid_data],
            valid_names=["train", "valid"],
            callbacks=callbacks
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
        if type == 2:
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



def select_features_by_importance(model, feature_names, threshold_ratio=0.01):
    """
    LightGBM の特徴量重要度を使って重要な特徴量だけを選択する。
    threshold_ratio: 重要度の合計に対する割合（例: 0.01 = 上位1%）
    """
    importance = model.feature_importance()
    total_importance = sum(importance)

    selected = []
    for name, imp in zip(feature_names, importance):
        if imp / total_importance >= threshold_ratio:
            selected.append(name)

    return selected


def auto_feature_selection(all_df, target_name, base_features, threshold_ratio=0.01, is_classification=False):
    """
    目的変数ごとに特徴量自動選択を行う。
    base_features: 最初に使う特徴量リスト
    threshold_ratio: 重要度の閾値（低いほど特徴量が減る）
    """
    #print(f"\n=== Auto Feature Selection for {target_name} ===")

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
        num_boost_round=500,
        valid_sets=[valid_data],
        valid_names=["valid"],
        callbacks=[lgb.early_stopping(stopping_rounds=50)]
    )

    # 特徴量選択
    selected_features = select_features_by_importance(
        model,
        model.feature_name(),
        threshold_ratio=threshold_ratio
    )

    disp = f"Selected Features ({len(selected_features)}):"
    for feat in selected_features:
        disp += f"{feat}, "
    print(disp)

    return selected_features




def _normalize_date_idx(date_idx):
    """dateIdx 指定を正規化して、1つでも複数でも扱えるようにする。"""
    if date_idx is None:
        return None
    if isinstance(date_idx, (list, tuple, set, pd.Index)):
        values = list(date_idx)
    else:
        values = [date_idx]
    return {pd.to_numeric(value, errors="raise") for value in values}


def predict_and_save(type, input_csv, output_csv=None, model_dir=".", date_idx=None):
    """保存済みモデルでCSVを予測し、予測列を末尾に追加して保存する。"""
    input_path = Path(input_csv)

    # ★ output_csv が指定されていなければ input_csv を上書きする
    if output_csv is None:
        output_path = input_path  # ←ここを変更
    else:
        output_path = Path(output_csv)

    df = pd.read_csv(input_path)
    if "dateIdx" not in df.columns:
        raise ValueError(f"{input_csv} に dateIdx 列がありません")

    df = add_technical_features(df)
    model_dir = Path(model_dir)

    target_date_idx = _normalize_date_idx(date_idx)
    target_mask = None if target_date_idx is None else df["dateIdx"].isin(target_date_idx)

    targets = ["futureStability", "futureUp", "futureFall","nextHigh", "nextLow"]
    if type == 2:
        #targets = ["futureBenefit","futureBenefitDate"]
        targets = ["futureBenefit"]
    for t in targets:
        is_classification = type == 2 and t == "futureBenefit"
        base_features = ALL_FEATURES.copy()
        if type == 2:
            base_features += PRED_FEATURES

        missing_features = [feature for feature in base_features if feature not in df.columns]
        if missing_features:
            raise ValueError(f"{input_csv} に必要な特徴量列がありません: {missing_features}")

        model_path = Path(f"{MODEL_BASE}/{t}model.txt")
        if not model_path.exists():
            raise FileNotFoundError(f"保存済みモデルが見つかりません: {model_path}")

        model = lgb.Booster(model_file=str(model_path))

        # 特徴量リストを読み込む
        with open(Path(MODEL_BASE) / f"{t}_features.txt") as f:
            selected_features = [line.strip() for line in f.readlines()]

        feature_data = df[selected_features].apply(pd.to_numeric, errors="coerce")
        if target_mask is not None:
            feature_data = feature_data.loc[target_mask]
            if feature_data.empty:
                continue
        pred = model.predict(feature_data)
        if is_classification:
            pred = pred.argmax(axis=1)
            pred = pred + 1

        pred_values = pred.to_numpy() if hasattr(pred, "to_numpy") else pred
        if target_mask is not None:
            if f"predicted_{t}" not in df.columns:
                df[f"predicted_{t}"] = pd.NA
            df.loc[target_mask, f"predicted_{t}"] = pred_values
        else:
            df[f"predicted_{t}"] = pred_values

    # ★ input_csv を上書き保存
    df.to_csv(output_path, index=False)
    return output_path


def predict_all(type, output_dir=None, model_dir=".", date_idx=None):
    """TechnicalScoreフォルダ内の全CSVを保存済みモデルで予測する。"""
    files = glob.glob(TECHNICAL_SCORE_GLOB)
    if not files:
        raise FileNotFoundError(f"予測対象のCSVが見つかりません: {TECHNICAL_SCORE_GLOB}")

    if output_dir is None:
        output_dir = Path(files[0]).parent #/ "predicted"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = []
    for input_csv in files:
        output_csv = output_dir / Path(input_csv).name
        output_paths.append(predict_and_save(type, input_csv, output_csv, model_dir, date_idx=date_idx))
    return output_paths

def add_technical_features(df):
    # ROC
    df["ROC5"] = df["close"].pct_change(5)
    df["ROC10"] = df["close"].pct_change(10)
    df["ROC20"] = df["close"].pct_change(20)

    # EMA
    df["EMA5"] = df["close"].ewm(span=5).mean()
    df["EMA20"] = df["close"].ewm(span=20).mean()
    df["EMA60"] = df["close"].ewm(span=60).mean()

    # EMA乖離率
    df["EMA5_div"] = (df["close"] - df["EMA5"]) / df["EMA5"]
    df["EMA20_div"] = (df["close"] - df["EMA20"]) / df["EMA20"]
    df["EMA60_div"] = (df["close"] - df["EMA60"]) / df["EMA60"]

    # MACD
    df["EMA12"] = df["close"].ewm(span=12).mean()
    df["EMA26"] = df["close"].ewm(span=26).mean()
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["MACD_signal"] = df["MACD"].ewm(span=9).mean()
    df["MACD_diff"] = df["MACD"] - df["MACD_signal"]

    # RSI
    diff = df["close"].diff()
    gain = diff.clip(lower=0)
    loss = -diff.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # RSI変化量
    df["RSI_delta"] = df["RSI"].diff()

    return df

def main():
    parser = argparse.ArgumentParser(description="テクニカル分析モデルの保存と予測")
    parser.add_argument("action", choices=["save", "predict"], help="実行する処理")
    parser.add_argument("type", type=int, help="タイプ")
    parser.add_argument("--date-idx", nargs="*", type=int, default=None, help="予測対象の dateIdx を指定する。未指定時は全行を対象にする")
    parser.add_argument("--output-dir",default=None,help="predict時の出力先フォルダ（省略時: TechnicalScore/predicted）")
    parser.add_argument("--model-dir",default=".",help="モデルファイルのフォルダ（省略時: カレントフォルダ）")
    args = parser.parse_args()

    if args.action == "save":
        saveModel(args.type)
    else:
        predict_all(args.type, args.output_dir, args.model_dir, date_idx=args.date_idx)


if __name__ == "__main__":
    main()
