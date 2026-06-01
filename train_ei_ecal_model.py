"""Train Ecal(Ei, index_i) from extracted test_6D input.

The target Ecal is first calculated from per-detector linear calibration
coefficients:

    Ecal = intercept[index_i] + slope[index_i] * Ei

Then a linear model is trained to predict Ecal from Ei and index_i. The script
also writes ROOT histograms for raw Ei and predicted Ecal.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from tqdm import tqdm


DEFAULT_COEFFICIENTS = Path("labr_energy_calibration_coefficients.csv")
TEST_COLUMNS = ["Ei", "index_i"]


def import_root():
    try:
        import ROOT
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "This script needs PyROOT to write/draw ROOT histograms. On the "
            "analysis server, load the ROOT environment/module before running it."
        ) from exc
    return ROOT


def load_input_dataframe(path: Path, use_filtered: bool) -> pd.DataFrame:
    if path.suffix.lower() == ".pkl":
        loaded = pd.read_pickle(path)
        if isinstance(loaded, dict):
            key = "df_filtered" if use_filtered else "df"
            if key not in loaded:
                raise KeyError(f"{path} does not contain {key!r}")
            df = loaded[key]
        else:
            df = loaded
    else:
        df = pd.read_csv(path)

    if "i" in df.columns and "index_i" not in df.columns:
        df = df.rename(columns={"i": "index_i"})

    missing = [column for column in TEST_COLUMNS if column not in df.columns]
    if missing:
        raise KeyError(f"{path} is missing required columns: {missing}")

    return df[TEST_COLUMNS].copy()


def load_calibration_coefficients(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Calibration coefficient file was not found: {path}")

    intercepts = np.zeros(18, dtype=float)
    slopes = np.zeros(18, dtype=float)
    has_coeff = np.zeros(18, dtype=bool)

    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            detector = int(row["detector"])
            if detector < 0 or detector >= 18:
                continue
            intercepts[detector] = float(row["intercept"])
            slopes[detector] = float(row["slope"])
            has_coeff[detector] = True

    if not np.any(has_coeff):
        raise ValueError(f"No detector coefficients were found in {path}")
    return intercepts, slopes, has_coeff


def apply_calibration(
    df: pd.DataFrame,
    intercepts: np.ndarray,
    slopes: np.ndarray,
    has_coeff: np.ndarray,
) -> pd.DataFrame:
    working = df.copy()
    working["index_i"] = working["index_i"].astype(int)
    valid_detector = working["index_i"].between(0, 17)
    valid_coeff = valid_detector & working["index_i"].map(lambda det: bool(has_coeff[det]))
    valid_energy = np.isfinite(working["Ei"].to_numpy(dtype=float)) & (working["Ei"] > 0)
    working = working[valid_coeff & valid_energy].copy()

    detectors = working["index_i"].to_numpy(dtype=int)
    raw_energy = working["Ei"].to_numpy(dtype=float)
    working["Ecal"] = intercepts[detectors] + slopes[detectors] * raw_energy
    working = working[np.isfinite(working["Ecal"]) & (working["Ecal"] > 0)].copy()
    return working


def make_feature_matrix(
    raw_energy: np.ndarray,
    detector_index: np.ndarray,
    detectors: np.ndarray,
) -> np.ndarray:
    features = np.zeros((raw_energy.size, detectors.size * 2), dtype=float)
    for out_index, detector in enumerate(detectors):
        mask = detector_index == detector
        features[mask, 2 * out_index] = 1.0
        features[mask, 2 * out_index + 1] = raw_energy[mask]
    return features


def train_model(df: pd.DataFrame) -> dict[str, Any]:
    detectors = np.array(sorted(df["index_i"].unique()), dtype=int)
    raw_energy = df["Ei"].to_numpy(dtype=float)
    detector_index = df["index_i"].to_numpy(dtype=int)
    target = df["Ecal"].to_numpy(dtype=float)
    features = make_feature_matrix(raw_energy, detector_index, detectors)

    model = LinearRegression(fit_intercept=False)
    model.fit(features, target)
    predicted = model.predict(features)
    mse = float(np.mean((target - predicted) ** 2))

    return {
        "model": model,
        "detectors": detectors,
        "mse": mse,
        "feature_description": "per-detector intercept and Ei slope columns",
    }


def predict_ecal(model_bundle: dict[str, Any], df: pd.DataFrame) -> np.ndarray:
    raw_energy = df["Ei"].to_numpy(dtype=float)
    detector_index = df["index_i"].to_numpy(dtype=int)
    features = make_feature_matrix(raw_energy, detector_index, model_bundle["detectors"])
    return model_bundle["model"].predict(features)


def binned_range(values: np.ndarray, bin_width: float) -> tuple[int, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 1, 0.0, bin_width
    low = np.floor(values.min() / bin_width) * bin_width
    high = np.ceil(values.max() / bin_width) * bin_width
    if high <= low:
        high = low + bin_width
    bins = max(1, int(round((high - low) / bin_width)))
    return bins, float(low), float(low + bins * bin_width)


def make_root_histogram(root: Any, name: str, title: str, counts: np.ndarray, low: float, high: float):
    hist = root.TH1D(name, title, len(counts), low, high)
    for index, count in enumerate(counts, start=1):
        hist.SetBinContent(index, float(count))
    return hist


def fill_test_tree(root: Any, output_file: Any, df: pd.DataFrame) -> None:
    tree = root.TTree("EiEcalTestData", "EiEcalTestData")
    ei = np.zeros(1, dtype=float)
    index_i = np.zeros(1, dtype=np.int32)
    ecal = np.zeros(1, dtype=float)
    ecal_pred = np.zeros(1, dtype=float)

    tree.Branch("Ei", ei, "Ei/D")
    tree.Branch("index_i", index_i, "index_i/I")
    tree.Branch("Ecal", ecal, "Ecal/D")
    tree.Branch("Ecal_pred", ecal_pred, "Ecal_pred/D")

    for row in df.itertuples(index=False):
        ei[0] = float(row.Ei)
        index_i[0] = int(row.index_i)
        ecal[0] = float(row.Ecal)
        ecal_pred[0] = float(row.Ecal_pred)
        tree.Fill()

    output_file.cd()
    tree.Write()


def write_histograms(
    df: pd.DataFrame,
    raw_bin_width: float,
    ecal_bin_width: float,
    output_path: Path,
    png_dir: Path | None,
) -> None:
    root = import_root()
    raw_values = df["Ei"].to_numpy(dtype=float)
    ecal_values = df["Ecal"].to_numpy(dtype=float)
    ecal_pred_values = df["Ecal_pred"].to_numpy(dtype=float)
    detector_values = df["index_i"].to_numpy(dtype=int)

    raw_bins, raw_low, raw_high = binned_range(raw_values, raw_bin_width)
    ecal_bins, ecal_low, ecal_high = binned_range(ecal_values, ecal_bin_width)
    ecal_pred_bins, ecal_pred_low, ecal_pred_high = binned_range(
        ecal_pred_values,
        ecal_bin_width,
    )
    raw_counts = np.histogram(raw_values, bins=raw_bins, range=(raw_low, raw_high))[0]
    ecal_counts = np.histogram(ecal_values, bins=ecal_bins, range=(ecal_low, ecal_high))[0]
    ecal_pred_counts = np.histogram(
        ecal_pred_values,
        bins=ecal_pred_bins,
        range=(ecal_pred_low, ecal_pred_high),
    )[0]

    raw_hist = make_root_histogram(
        root,
        "Ei_raw",
        f"Raw Ei;Ei;Counts",
        raw_counts,
        raw_low,
        raw_high,
    )
    ecal_hist = make_root_histogram(
        root,
        "Ei_ecal",
        f"Calibrated Ei;Ecal;Counts",
        ecal_counts,
        ecal_low,
        ecal_high,
    )
    ecal_pred_hist = make_root_histogram(
        root,
        "Ei_ecal_pred",
        f"Predicted calibrated Ei;Ecal_pred;Counts",
        ecal_pred_counts,
        ecal_pred_low,
        ecal_pred_high,
    )

    det_vs_eraw = root.TH2D(
        "detector_index_vs_Ei_raw",
        "Detector index vs raw Ei;Detector index;Ei",
        18,
        -0.5,
        17.5,
        raw_bins,
        raw_low,
        raw_high,
    )
    det_vs_ecal = root.TH2D(
        "detector_index_vs_Ecal",
        "Detector index vs calibrated Ei;Detector index;Ecal",
        18,
        -0.5,
        17.5,
        ecal_bins,
        ecal_low,
        ecal_high,
    )
    det_vs_ecal_pred = root.TH2D(
        "detector_index_vs_Ecal_pred",
        "Detector index vs predicted calibrated Ei;Detector index;Ecal_pred",
        18,
        -0.5,
        17.5,
        ecal_pred_bins,
        ecal_pred_low,
        ecal_pred_high,
    )

    for detector, raw_energy, ecal, ecal_pred in zip(
        detector_values,
        raw_values,
        ecal_values,
        ecal_pred_values,
    ):
        det_vs_eraw.Fill(float(detector), float(raw_energy))
        det_vs_ecal.Fill(float(detector), float(ecal))
        det_vs_ecal_pred.Fill(float(detector), float(ecal_pred))

    output_file = root.TFile(str(output_path), "RECREATE")
    fill_test_tree(root, output_file, df)
    raw_hist.Write()
    ecal_hist.Write()
    ecal_pred_hist.Write()
    det_vs_eraw.Write()
    det_vs_ecal.Write()
    det_vs_ecal_pred.Write()
    output_file.Close()

    if png_dir:
        png_dir.mkdir(parents=True, exist_ok=True)
        canvas = root.TCanvas("canvas", "canvas", 1000, 700)
        raw_hist.Draw()
        canvas.SaveAs(str(png_dir / "Ei_raw.png"))
        canvas.Clear()
        ecal_hist.Draw()
        canvas.SaveAs(str(png_dir / "Ei_ecal.png"))
        canvas.Clear()
        ecal_pred_hist.Draw()
        canvas.SaveAs(str(png_dir / "Ei_ecal_pred.png"))
        canvas.Clear()
        det_vs_eraw.Draw("COLZ")
        canvas.SaveAs(str(png_dir / "detector_index_vs_Ei_raw.png"))
        canvas.Clear()
        det_vs_ecal.Draw("COLZ")
        canvas.SaveAs(str(png_dir / "detector_index_vs_Ecal.png"))
        canvas.Clear()
        det_vs_ecal_pred.Draw("COLZ")
        canvas.SaveAs(str(png_dir / "detector_index_vs_Ecal_pred.png"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Ecal(Ei, index_i) and draw raw/calibrated histograms."
    )
    parser.add_argument(
        "input_data",
        type=Path,
        help="test_6D input .pkl or .csv, for example test_6D_input.pkl.",
    )
    parser.add_argument(
        "--coefficients",
        type=Path,
        default=DEFAULT_COEFFICIENTS,
        help=f"Calibration coefficients CSV. Default: {DEFAULT_COEFFICIENTS}.",
    )
    parser.add_argument(
        "--use-filtered",
        action="store_true",
        help="For .pkl dict input, use df_filtered instead of df.",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("ei_to_ecal_model.joblib"),
        help="Output joblib model path.",
    )
    parser.add_argument(
        "--hist-output",
        type=Path,
        default=Path("ei_ecal_histograms.root"),
        help="Output ROOT file with Ei and Ecal histograms.",
    )
    parser.add_argument("--raw-bin-width", type=float, default=10.0)
    parser.add_argument("--ecal-bin-width", type=float, default=1.0)
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.1,
        help="Fraction of rows held out as test data. Default: 0.1 = 10%%.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--png-dir", type=Path, help="Optional PNG output directory.")
    parser.add_argument(
        "--save-data",
        type=Path,
        help="Optional CSV with Ei, index_i, Ecal target, and Ecal prediction.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.raw_bin_width <= 0:
        raise ValueError("--raw-bin-width must be greater than 0")
    if args.ecal_bin_width <= 0:
        raise ValueError("--ecal-bin-width must be greater than 0")
    if not 0 < args.test_size < 1:
        raise ValueError("--test-size must be greater than 0 and less than 1")

    with tqdm(total=6, desc="Workflow", unit="step") as pbar:
        df = load_input_dataframe(args.input_data, args.use_filtered)
        pbar.set_postfix_str("loaded input")
        pbar.update(1)

        intercepts, slopes, has_coeff = load_calibration_coefficients(args.coefficients)
        pbar.set_postfix_str("loaded coefficients")
        pbar.update(1)

        calibrated = apply_calibration(df, intercepts, slopes, has_coeff)
        if calibrated.empty:
            raise ValueError("No valid Ei/index_i rows remained after applying coefficients")
        pbar.set_postfix_str("applied calibration")
        pbar.update(1)

        train_df, test_df = train_test_split(
            calibrated,
            test_size=args.test_size,
            random_state=args.random_state,
            shuffle=True,
        )
        pbar.set_postfix_str("split train/test")
        pbar.update(1)

        model_bundle = train_model(train_df)
        test_df = test_df.copy()
        test_df["Ecal_pred"] = predict_ecal(model_bundle, test_df)
        test_mse = float(np.mean((test_df["Ecal"] - test_df["Ecal_pred"]) ** 2))
        joblib.dump(model_bundle, args.model_output)
        pbar.set_postfix_str("trained model")
        pbar.update(1)

        write_histograms(
            test_df,
            args.raw_bin_width,
            args.ecal_bin_width,
            args.hist_output,
            args.png_dir,
        )
        pbar.set_postfix_str("wrote ROOT output")
        pbar.update(1)

    if args.save_data:
        test_df.to_csv(args.save_data, index=False)

    train_percent = (1.0 - args.test_size) * 100.0
    test_percent = args.test_size * 100.0
    print(f"Training/test split: {train_percent:.1f}% train, {test_percent:.1f}% test")
    print(f"Rows after calibration cuts: {len(calibrated)}")
    print(f"Training rows: {len(train_df)}")
    print(f"Test rows written to ROOT: {len(test_df)}")
    print(f"Detectors used: {', '.join(map(str, model_bundle['detectors']))}")
    print(f"Training MSE: {model_bundle['mse']:.6g}")
    print(f"Test MSE: {test_mse:.6g}")
    print(f"Wrote {args.model_output}")
    print(f"Wrote {args.hist_output}")
    if args.save_data:
        print(f"Wrote {args.save_data}")


if __name__ == "__main__":
    main()
