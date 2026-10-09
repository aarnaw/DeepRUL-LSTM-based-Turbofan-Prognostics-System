import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

SEQ_LEN = 30
RUL_CAP = 125


def load_fd004() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cols = ["unit_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"] + [f"sensor_{i}" for i in range(1, 22)]
    train = pd.read_csv(RAW_DIR / "train_FD004.txt", sep=r"\s+", header=None, names=cols)
    test = pd.read_csv(RAW_DIR / "test_FD004.txt", sep=r"\s+", header=None, names=cols)
    rul = pd.read_csv(RAW_DIR / "RUL_FD004.txt", sep=r"\s+", header=None, names=["final_rul"])
    return train, test, rul


def add_labels_and_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    drop_cols = [
        "op_setting_3",
        "sensor_1", "sensor_5", "sensor_6", "sensor_10", "sensor_16", "sensor_18", "sensor_19",
    ]
    train_df = train_df.drop(columns=drop_cols).copy()
    test_df = test_df.drop(columns=drop_cols).copy()

    sensor_cols = [c for c in train_df.columns if c.startswith("sensor_")]

    train_df["cycle_norm"] = train_df["cycle"] / train_df.groupby("unit_id")["cycle"].transform("max")
    test_df["cycle_norm"] = test_df["cycle"] / test_df.groupby("unit_id")["cycle"].transform("max")

    max_cycle = train_df.groupby("unit_id")["cycle"].max().rename("max_cycle")
    train_df = train_df.join(max_cycle, on="unit_id")
    train_df["RUL_raw"] = train_df["max_cycle"] - train_df["cycle"]
    train_df["RUL"] = train_df["RUL_raw"].clip(upper=RUL_CAP)
    train_df = train_df.drop(columns=["max_cycle"])

    feature_cols = sensor_cols + ["cycle_norm"]
    return train_df, test_df, sensor_cols, feature_cols


def scale_features(train_df: pd.DataFrame, test_df: pd.DataFrame, sensor_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    scaler = MinMaxScaler()
    train_df[sensor_cols] = scaler.fit_transform(train_df[sensor_cols].astype(float))
    test_df[sensor_cols] = scaler.transform(test_df[sensor_cols].astype(float))

    with open(PROCESSED_DIR / "feature_scaler.pkl", "wb") as fp:
        pickle.dump(scaler, fp)

    return train_df, test_df


def build_train_sequences(df: pd.DataFrame, feature_cols: list[str], seq_len: int):
    X, y, unit_ids = [], [], []
    for unit_id in df["unit_id"].unique():
        engine = df[df["unit_id"] == unit_id].sort_values("cycle")
        features = engine[feature_cols].to_numpy(dtype=np.float32)
        rul = engine["RUL"].to_numpy(dtype=np.float32)

        for i in range(len(features) - seq_len + 1):
            X.append(features[i:i + seq_len])
            y.append(rul[i + seq_len - 1])
            unit_ids.append(unit_id)

    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32), np.asarray(unit_ids, dtype=np.int32)


def build_test_last_sequences(df: pd.DataFrame, rul_table: pd.DataFrame, feature_cols: list[str], seq_len: int):
    X_test, y_test, ids = [], [], []
    final_rul_map = {i + 1: float(v) for i, v in enumerate(rul_table["final_rul"].tolist())}

    for unit_id in df["unit_id"].unique():
        engine = df[df["unit_id"] == unit_id].sort_values("cycle")
        features = engine[feature_cols].to_numpy(dtype=np.float32)

        if len(features) < seq_len:
            pad = np.repeat(features[[0]], seq_len - len(features), axis=0)
            features = np.vstack([pad, features])

        X_test.append(features[-seq_len:])
        y_test.append(min(final_rul_map[unit_id], RUL_CAP))
        ids.append(unit_id)

    return np.asarray(X_test, dtype=np.float32), np.asarray(y_test, dtype=np.float32), np.asarray(ids, dtype=np.int32)


def sequence_balance(X: np.ndarray, y: np.ndarray, unit_ids: np.ndarray):
    high_mask = y == RUL_CAP
    high_idx = np.where(high_mask)[0]
    low_idx = np.where(~high_mask)[0]

    rng = np.random.default_rng(42)
    keep_high_n = int(len(high_idx) * 0.3)
    keep_high_idx = rng.choice(high_idx, size=keep_high_n, replace=False) if keep_high_n > 0 else high_idx

    keep_idx = np.concatenate([low_idx, keep_high_idx])
    rng.shuffle(keep_idx)
    return X[keep_idx], y[keep_idx], unit_ids[keep_idx], len(high_idx), int(np.sum(y[keep_idx] == RUL_CAP))


def build_test_trajectories(test_df: pd.DataFrame, rul_df: pd.DataFrame, feature_cols: list[str]):
    final_rul_map = {i + 1: float(v) for i, v in enumerate(rul_df["final_rul"].tolist())}
    trajectories = {}

    for unit_id in test_df["unit_id"].unique():
        engine = test_df[test_df["unit_id"] == unit_id].sort_values("cycle").copy()
        max_obs = int(engine["cycle"].max())
        engine["actual_rul"] = ((max_obs - engine["cycle"]) + final_rul_map[unit_id]).clip(upper=RUL_CAP)

        features = engine[feature_cols].to_numpy(dtype=np.float32)
        actual = engine["actual_rul"].to_numpy(dtype=np.float32)
        cycles = engine["cycle"].to_numpy(dtype=np.int32)

        seqs, actuals, cycle_list = [], [], []
        for i in range(len(engine)):
            window = features[max(0, i - SEQ_LEN + 1): i + 1]
            if len(window) < SEQ_LEN:
                pad = np.repeat(window[[0]], SEQ_LEN - len(window), axis=0)
                window = np.vstack([pad, window])
            seqs.append(window.astype(np.float32))
            actuals.append(float(actual[i]))
            cycle_list.append(int(cycles[i]))

        trajectories[int(unit_id)] = {
            "cycles": cycle_list,
            "actual_rul": actuals,
            "sequences": np.asarray(seqs, dtype=np.float32),
            "final_rul": float(min(final_rul_map[unit_id], RUL_CAP)),
        }

    return trajectories


def main() -> None:
    train_df, test_df, rul_df = load_fd004()
    train_df, test_df, sensor_cols, feature_cols = add_labels_and_features(train_df, test_df)
    train_df, test_df = scale_features(train_df, test_df, sensor_cols)

    X_train, y_train, train_unit_ids = build_train_sequences(train_df, feature_cols, SEQ_LEN)
    X_train, y_train, train_unit_ids, high_before, high_after = sequence_balance(X_train, y_train, train_unit_ids)

    X_test, y_test, test_unit_ids = build_test_last_sequences(test_df, rul_df, feature_cols, SEQ_LEN)

    np.save(PROCESSED_DIR / "X_train_sequences.npy", X_train)
    np.save(PROCESSED_DIR / "y_train_sequences.npy", y_train)
    np.save(PROCESSED_DIR / "train_sequence_unit_ids.npy", train_unit_ids)
    np.save(PROCESSED_DIR / "X_test_last.npy", X_test)
    np.save(PROCESSED_DIR / "y_test_last.npy", y_test)
    np.save(PROCESSED_DIR / "test_unit_ids.npy", test_unit_ids)

    with open(PROCESSED_DIR / "feature_columns.json", "w", encoding="utf-8") as fp:
        json.dump(feature_cols, fp, indent=2)

    with open(PROCESSED_DIR / "dataset_config.json", "w", encoding="utf-8") as fp:
        json.dump({"dataset": "FD004", "seq_len": SEQ_LEN, "rul_cap": RUL_CAP}, fp, indent=2)

    trajectories = build_test_trajectories(test_df, rul_df, feature_cols)
    with open(PROCESSED_DIR / "test_engine_trajectories.pkl", "wb") as fp:
        pickle.dump(trajectories, fp)

    sample_count = min(100, len(X_test))
    with open(PROCESSED_DIR / "test_sequences_sample.json", "w", encoding="utf-8") as fp:
        json.dump({"samples": X_test[:sample_count].tolist()}, fp)

    print("FD004 preprocessing complete")
    print(f"X_train: {X_train.shape} | y_train: {y_train.shape}")
    print(f"X_test : {X_test.shape} | y_test : {y_test.shape}")
    print(f"RUL=125 sequences before/after balancing: {high_before}/{high_after}")
    print(f"Feature count: {len(feature_cols)}")


if __name__ == "__main__":
    main()
