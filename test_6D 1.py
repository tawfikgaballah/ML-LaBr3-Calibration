# Author: Tawfik Gaballah
# GitHub: tawfikgaballah
# Project: ML-LaBr3-Calibration

"""Train and evaluate the 6D LaBr timing correction model.

The model learns measured time difference as a function of calibrated energies
and detector indices:

    tdiff = f(Ei, index_i, Ej, index_j)

The corrected timing residual written to the output ROOT file is:

    T_Diff_Corrected = T_Diff - T_pred
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm


FEATURE_COLUMNS = ["Ei", "index_i", "Ej", "index_j"]
DATA_COLUMNS = FEATURE_COLUMNS + ["tdiff"]
TIME_BIN_WIDTH = 0.01
ENERGY_BIN_WIDTH = 10.0


class Log1pTransformer:
    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.log1p(X)


def run_label_from_path(path: str | Path) -> str:
    path = Path(path)
    match = re.search(r"(run[-_]?\d+)", path.name, flags=re.IGNORECASE)
    if match:
        return match.group(1).replace("_", "-")
    stem = path.stem
    if stem.startswith("test_6D_input_"):
        stem = stem[len("test_6D_input_"):]
    return stem


def apply_default_output_paths(args: argparse.Namespace) -> None:
    run_label = run_label_from_path(args.input_data)
    if args.model_output is None:
        args.model_output = f"trained_model_{run_label}.joblib"
    if args.root_output is None:
        args.root_output = f"test_6D_output_{run_label}.root"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/evaluate the 6D timing model from extracted pandas input."
    )
    parser.add_argument(
        "input_data",
        nargs="?",
        default="test_6D_input.pkl",
        help="Extractor output .pkl, or CSV with Ei,index_i,Ej,index_j,tdiff columns.",
    )
    parser.add_argument(
        "--model-output",
        help="Path where the trained joblib pipeline is written. Default includes run label.",
    )
    parser.add_argument(
        "--root-output",
        help="Path where the diagnostic ROOT file is written. Default includes run label.",
    )
    parser.add_argument("--model-plot", help="Optional path for a Keras model diagram.")
    parser.add_argument("--tdiff-min", type=float, default=-20.0)
    parser.add_argument("--tdiff-max", type=float, default=20.0)
    parser.add_argument(
        "--data-fraction",
        type=float,
        default=1.0,
        help=(
            "Fraction of filtered rows to use before the train/test split. "
            "Use 1.0 for all rows, 0.1 for 10 percent, etc."
        ),
    )
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--huber-delta", type=float, default=0.2)
    parser.add_argument("--prompt-low", type=float, default=494.55)
    parser.add_argument("--prompt-high", type=float, default=527.45)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.tdiff_min >= args.tdiff_max:
        raise ValueError("--tdiff-min must be less than --tdiff-max")
    if not 0 < args.data_fraction <= 1:
        raise ValueError("--data-fraction must be greater than 0 and less than or equal to 1")
    if not 0 < args.test_size < 1:
        raise ValueError("--test-size must be between 0 and 1")
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be greater than 0")
    if args.huber_delta <= 0:
        raise ValueError("--huber-delta must be greater than 0")


def load_dataframe(input_data: str | Path) -> pd.DataFrame:
    input_path = Path(input_data)
    if input_path.suffix.lower() == ".pkl":
        loaded = pd.read_pickle(input_path)
        df = loaded.get("df") if isinstance(loaded, dict) else loaded
        if df is None:
            raise KeyError(f"{input_path} does not contain a 'df' dataframe")
    else:
        df = pd.read_csv(input_path)

    missing = [name for name in DATA_COLUMNS if name not in df.columns]
    if missing:
        raise KeyError(f"{input_path} is missing required columns: {missing}")
    return df[DATA_COLUMNS].copy()


def filter_test_6d_df(df: pd.DataFrame, tdiff_min: float, tdiff_max: float) -> pd.DataFrame:
    mask = (
        df["tdiff"].between(tdiff_min, tdiff_max)
        & (df["index_i"].astype(int) != df["index_j"].astype(int))
    )
    return df.loc[mask, DATA_COLUMNS].copy()


def sample_data_fraction(
    df_filtered: pd.DataFrame,
    data_fraction: float,
    random_state: int,
) -> pd.DataFrame:
    if data_fraction >= 1.0:
        return df_filtered.copy()
    sampled = df_filtered.sample(frac=data_fraction, random_state=random_state)
    return sampled.sort_index().copy()


def split_training_data(
    df_filtered: pd.DataFrame,
    test_size: float,
    random_state: int,
):
    from sklearn.model_selection import train_test_split

    X = df_filtered[FEATURE_COLUMNS].copy()
    y = df_filtered["tdiff"].copy()
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def batch_count(row_count: int, batch_size: int, epochs: int = 1) -> int:
    if row_count <= 0:
        return 0
    return int(np.ceil(row_count / batch_size)) * epochs


def log_step(message: str) -> None:
    print(f"[test_6D] {message}", flush=True)


def make_batch_progress_callback(description: str, total_batches: int) -> Any:
    from tensorflow.keras.callbacks import Callback

    class TqdmBatchProgress(Callback):
        def __init__(self):
            super().__init__()
            self.progress = None

        def _open(self):
            self.progress = tqdm(
                total=total_batches,
                desc=description,
                unit="batch",
                ascii=True,
                dynamic_ncols=True,
            )

        def _update(self):
            if self.progress is not None:
                self.progress.update(1)

        def _close(self):
            if self.progress is not None:
                self.progress.close()
                self.progress = None

        def on_train_begin(self, logs=None):
            self._open()

        def on_train_batch_end(self, batch, logs=None):
            self._update()

        def on_train_end(self, logs=None):
            self._close()

        def on_test_begin(self, logs=None):
            self._open()

        def on_test_batch_end(self, batch, logs=None):
            self._update()

        def on_test_end(self, logs=None):
            self._close()

        def on_predict_begin(self, logs=None):
            self._open()

        def on_predict_batch_end(self, batch, logs=None):
            self._update()

        def on_predict_end(self, logs=None):
            self._close()

    return TqdmBatchProgress()


def dense_one_hot_encoder() -> Any:
    from sklearn.preprocessing import OneHotEncoder

    try:
        return OneHotEncoder(categories="auto", sparse_output=False)
    except TypeError:
        return OneHotEncoder(categories="auto", sparse=False)


def build_preprocessor() -> Any:
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler

    return ColumnTransformer(
        transformers=[
            ("log_transform", Log1pTransformer(), [0, 2]),
            ("onehot", dense_one_hot_encoder(), [1, 3]),
            ("scale", StandardScaler(), [0, 2]),
        ],
        remainder="passthrough",
    )


def build_keras_model(learning_rate: float, huber_delta: float) -> Any:
    import tensorflow as tf
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.losses import Huber
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam

    tf.get_logger().setLevel("ERROR")
    model = Sequential(
        [
            Dense(128, activation="relu"),
            Dense(64, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=Huber(delta=huber_delta))
    return model


def build_pipeline(args: argparse.Namespace) -> tuple[Any, Any]:
    from sklearn.pipeline import Pipeline

    keras_model = build_keras_model(args.learning_rate, args.huber_delta)
    pipeline = Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("regressor", keras_model),
        ]
    )
    return pipeline, keras_model


def evaluate_model(
    pipeline: Any,
    keras_model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    batch_size: int,
) -> tuple[np.ndarray, float, float]:
    X_test_transformed = pipeline.named_steps["preprocessor"].transform(X_test)
    total_batches = batch_count(len(X_test), batch_size)
    score = keras_model.evaluate(
        X_test_transformed,
        y_test,
        batch_size=batch_size,
        verbose=0,
        callbacks=[make_batch_progress_callback("Evaluating test loss", total_batches)],
    )
    loss_value = score[0] if isinstance(score, (list, tuple)) else score
    y_pred = keras_model.predict(
        X_test_transformed,
        batch_size=batch_size,
        verbose=0,
        callbacks=[make_batch_progress_callback("Predicting test data", total_batches)],
    )
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    mse = float(np.mean((y_test.to_numpy(dtype=float) - y_pred) ** 2))
    return y_pred, float(loss_value), mse


def padded_range(values: np.ndarray, fallback=(-1.0, 1.0), padding_fraction=0.05):
    values = np.asarray(values, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return fallback
    low = float(values.min())
    high = float(values.max())
    if low == high:
        pad = max(abs(low) * padding_fraction, 1.0)
    else:
        pad = (high - low) * padding_fraction
    return low - pad, high + pad


def binned_axis(low: float, high: float, bin_width: float):
    bins = max(1, int(np.ceil((high - low) / bin_width)))
    return bins, low, low + bins * bin_width


def import_root():
    import ROOT

    return ROOT


def create_output_tree(ROOT: Any):
    tree = ROOT.TTree("TreeOutput", "TreeOutput")
    buffers = {
        "Ei": np.zeros(1, dtype=np.float64),
        "index_i": np.zeros(1, dtype=np.int32),
        "Ej": np.zeros(1, dtype=np.float64),
        "index_j": np.zeros(1, dtype=np.int32),
        "T_Diff_Corrected": np.zeros(1, dtype=np.float64),
        "T_Diff": np.zeros(1, dtype=np.float64),
        "T_pred": np.zeros(1, dtype=np.float64),
        "Dynode": np.zeros(1, dtype=np.float64),
    }
    tree.Branch("Ei", buffers["Ei"], "Ei/D")
    tree.Branch("index_i", buffers["index_i"], "index_i/I")
    tree.Branch("Ej", buffers["Ej"], "Ej/D")
    tree.Branch("index_j", buffers["index_j"], "index_j/I")
    tree.Branch("T_Diff_Corrected", buffers["T_Diff_Corrected"], "T_Diff_Corrected/D")
    tree.Branch("T_Diff", buffers["T_Diff"], "T_Diff/D")
    tree.Branch("T_pred", buffers["T_pred"], "T_pred/D")
    tree.Branch("Dynode", buffers["Dynode"], "Dynode/D")
    return tree, buffers


def create_histograms(ROOT: Any, X_test: pd.DataFrame, y_test: np.ndarray, corrected: np.ndarray):
    tdiff_corr_min, tdiff_corr_max = padded_range(corrected)
    tdiff_min, tdiff_max = padded_range(y_test)
    ej_min, ej_max = padded_range(X_test["Ej"].to_numpy(dtype=float), fallback=(0.0, 1.0))

    time_corr_bins, tdiff_corr_min, tdiff_corr_max = binned_axis(
        tdiff_corr_min,
        tdiff_corr_max,
        TIME_BIN_WIDTH,
    )
    time_bins, tdiff_min, tdiff_max = binned_axis(tdiff_min, tdiff_max, TIME_BIN_WIDTH)
    energy_bins, ej_min, ej_max = binned_axis(ej_min, ej_max, ENERGY_BIN_WIDTH)

    return {
        "Ej_Vs_T_Diff_Corrected": ROOT.TH2D(
            "Ej_Vs_T_Diff_Corrected",
            "Ej_Vs_T_Diff_Corrected;T_Diff_Corrected;Ej",
            time_corr_bins,
            tdiff_corr_min,
            tdiff_corr_max,
            energy_bins,
            ej_min,
            ej_max,
        ),
        "Ej_Vs_T_Diff": ROOT.TH2D(
            "Ej_Vs_T_Diff",
            "Ej_Vs_T_Diff;T_Diff;Ej",
            time_bins,
            tdiff_min,
            tdiff_max,
            energy_bins,
            ej_min,
            ej_max,
        ),
        "T_Diff_Counts_Vs_Detector_Pair": ROOT.TH2D(
            "T_Diff_Counts_Vs_Detector_Pair",
            "T_Diff_Counts_Vs_Detector_Pair;index_i;index_j",
            18,
            0,
            18,
            18,
            0,
            18,
        ),
        "Average_T_Diff_Corrected_Vs_Detector_Pair": ROOT.TH2D(
            "Average_T_Diff_Corrected_Vs_Detector_Pair",
            "Average_T_Diff_Corrected_Vs_Detector_Pair;index_i;index_j",
            18,
            0,
            18,
            18,
            0,
            18,
        ),
        "Prompt_Response_Corrected": ROOT.TH1D(
            "Prompt_Response_Corrected",
            "Prompt_Response_Corrected;T_Diff_Corrected;Counts",
            time_corr_bins,
            tdiff_corr_min,
            tdiff_corr_max,
        ),
        "Prompt_Response": ROOT.TH1D(
            "Prompt_Response",
            "Prompt_Response;T_Diff;Counts",
            time_bins,
            tdiff_min,
            tdiff_max,
        ),
    }


def fill_output(
    tree: Any,
    buffers: dict[str, np.ndarray],
    histograms: dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
    prompt_low: float,
    prompt_high: float,
) -> None:
    y_true = y_test.to_numpy(dtype=float)
    corrected = y_true - y_pred
    pair_counts = np.zeros((18, 18), dtype=float)
    pair_sum_corrected = np.zeros((18, 18), dtype=float)

    for row_index, row in enumerate(
        tqdm(
            X_test.itertuples(index=False),
            total=len(X_test),
            desc="Writing diagnostics",
            unit="rows",
            ascii=True,
            dynamic_ncols=True,
        )
    ):
        index_i = int(row.index_i)
        index_j = int(row.index_j)

        buffers["Ei"][0] = float(row.Ei)
        buffers["index_i"][0] = index_i
        buffers["Ej"][0] = float(row.Ej)
        buffers["index_j"][0] = index_j
        buffers["T_Diff"][0] = y_true[row_index]
        buffers["T_pred"][0] = y_pred[row_index]
        buffers["T_Diff_Corrected"][0] = corrected[row_index]
        buffers["Dynode"][0] = 0.0
        tree.Fill()

        histograms["Ej_Vs_T_Diff_Corrected"].Fill(corrected[row_index], float(row.Ej))
        histograms["Ej_Vs_T_Diff"].Fill(y_true[row_index], float(row.Ej))

        if 0 <= index_i < 18 and 0 <= index_j < 18:
            pair_counts[index_i, index_j] += 1.0
            pair_sum_corrected[index_i, index_j] += corrected[row_index]

        if (
            prompt_low <= float(row.Ei) <= prompt_high
            and prompt_low <= float(row.Ej) <= prompt_high
        ):
            histograms["Prompt_Response_Corrected"].Fill(corrected[row_index])
            histograms["Prompt_Response"].Fill(y_true[row_index])

    for index_i in range(18):
        for index_j in range(18):
            histograms["T_Diff_Counts_Vs_Detector_Pair"].SetBinContent(
                index_i + 1,
                index_j + 1,
                pair_counts[index_i, index_j],
            )
            if pair_counts[index_i, index_j] > 0:
                histograms["Average_T_Diff_Corrected_Vs_Detector_Pair"].SetBinContent(
                    index_i + 1,
                    index_j + 1,
                    pair_sum_corrected[index_i, index_j] / pair_counts[index_i, index_j],
                )


def write_root_output(
    output_path: str | Path,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
    prompt_low: float,
    prompt_high: float,
) -> None:
    y_true = y_test.to_numpy(dtype=float)
    corrected = y_true - y_pred
    ROOT = import_root()
    root_file = ROOT.TFile(str(output_path), "RECREATE")
    tree, buffers = create_output_tree(ROOT)
    histograms = create_histograms(ROOT, X_test, y_true, corrected)
    fill_output(tree, buffers, histograms, X_test, y_test, y_pred, prompt_low, prompt_high)

    root_file.cd()
    for histogram in histograms.values():
        histogram.Write()
    tree.Write()
    root_file.Close()


def print_input_summary(
    df: pd.DataFrame,
    df_filtered: pd.DataFrame,
    df_used: pd.DataFrame,
    data_fraction: float,
) -> None:
    removed_same_detector = int((df["index_i"].astype(int) == df["index_j"].astype(int)).sum())
    print(f"Rows loaded: {len(df)}")
    print(f"Rows with index_i == index_j removed by filter: {removed_same_detector}")
    print(f"Rows after tdiff and detector-pair filtering: {len(df_filtered)}")
    print(f"Rows used for train/test split: {len(df_used)} ({data_fraction:g} fraction)")
    if not df_used.empty:
        print("Filtered detector-pair counts:")
        counts = df_used.groupby(["index_i", "index_j"]).size().sort_index()
        print(counts.to_string())


def main() -> None:
    args = parse_args()
    validate_args(args)
    apply_default_output_paths(args)

    log_step(f"Loading input data from {args.input_data}")
    df = load_dataframe(args.input_data)
    log_step("Filtering tdiff range and removing index_i == index_j rows")
    df_filtered = filter_test_6d_df(df, args.tdiff_min, args.tdiff_max)
    if df_filtered.empty:
        raise ValueError("No rows remain after filtering tdiff and index_i != index_j")

    log_step(f"Sampling {args.data_fraction:g} of the filtered rows")
    df_used = sample_data_fraction(df_filtered, args.data_fraction, args.random_state)
    if df_used.empty:
        raise ValueError("No rows remain after applying --data-fraction")

    print_input_summary(df, df_filtered, df_used, args.data_fraction)
    log_step(f"Splitting data with test-size {args.test_size:g}")
    X_train, X_test, y_train, y_test = split_training_data(
        df_used,
        args.test_size,
        args.random_state,
    )
    log_step(f"Training rows: {len(X_train)}; test rows: {len(X_test)}")

    log_step("Building timing model")
    pipeline, keras_model = build_pipeline(args)
    if args.model_plot:
        from tensorflow.keras.utils import plot_model

        log_step(f"Writing model plot to {args.model_plot}")
        plot_model(keras_model, to_file=args.model_plot, show_shapes=True, show_layer_names=True)

    log_step(f"Training for {args.epochs} epoch(s) with batch size {args.batch_size}")
    pipeline.fit(
        X_train,
        y_train,
        regressor__batch_size=args.batch_size,
        regressor__epochs=args.epochs,
        regressor__verbose=0,
        regressor__callbacks=[
            make_batch_progress_callback(
                "Training model",
                batch_count(len(X_train), args.batch_size, args.epochs),
            )
        ],
    )

    log_step("Evaluating and predicting test rows")
    y_pred, loss_value, mse = evaluate_model(
        pipeline,
        keras_model,
        X_test,
        y_test,
        args.batch_size,
    )
    print(f"Test loss: {loss_value:.6g}")
    print(f"Mean squared error: {mse:.6g}")

    from joblib import dump

    log_step(f"Saving trained model to {args.model_output}")
    dump(pipeline, args.model_output)
    print(f"Wrote model: {args.model_output}")

    log_step(f"Writing ROOT diagnostics to {args.root_output}")
    write_root_output(
        args.root_output,
        X_test,
        y_test,
        y_pred,
        args.prompt_low,
        args.prompt_high,
    )
    print(f"Wrote ROOT diagnostics: {args.root_output}")
    print("Processing completed successfully.")


if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    main()
