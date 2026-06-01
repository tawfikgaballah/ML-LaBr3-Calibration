# Author: Tawfik Gaballah
# GitHub: tawfikgaballah
# Project: ML-LaBr3-Calibration

"""Apply a trained Ei -> Ecal model to raw LaBr energies in ROOT files."""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path
from typing import Any

import numpy as np
import uproot
from tqdm import tqdm


LABR_ENERGY_BRANCH = "rootout/labr/labr.energy[18]"


def import_root():
    try:
        import ROOT
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "This script needs PyROOT to write ROOT outputs. On the analysis "
            "server, load the ROOT environment/module before running it."
        ) from exc
    return ROOT


def import_joblib():
    try:
        import joblib
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "This script needs joblib to load the trained model. Install the "
            "requirements file in your active environment."
        ) from exc
    return joblib


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


def looks_like_number(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def percentage_to_fraction(value: float | None) -> float | None:
    if value is None:
        return None
    if value <= 0:
        raise ValueError("percentage must be greater than 0")
    if value <= 100:
        return value / 100.0
    raise ValueError("percentage must be at most 100")


def split_inputs_and_percentage(values: list[str]) -> tuple[list[str], float | None]:
    if not values:
        raise ValueError("at least one ROOT file or glob pattern is required")
    if looks_like_number(values[-1]):
        if len(values) == 1:
            raise ValueError("a ROOT file or glob pattern is required before percentage")
        return values[:-1], float(values[-1])
    return values, None


def expand_input_patterns(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        if any(char in pattern for char in "*?["):
            matches = sorted(glob.glob(pattern))
            if not matches:
                raise FileNotFoundError(f"No files matched pattern: {pattern}")
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(pattern))

    unique_paths: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique_paths.append(path)
    return unique_paths


def run_label_from_path(path: Path) -> str:
    match = re.search(r"(run[-_]?\d+)", path.name, flags=re.IGNORECASE)
    if match:
        return match.group(1).replace("_", "-")
    return path.stem


def run_label_from_paths(paths: list[Path]) -> str:
    labels: list[str] = []
    seen: set[str] = set()
    for path in paths:
        label = run_label_from_path(path)
        if label not in seen:
            seen.add(label)
            labels.append(label)

    if len(labels) == 1:
        return labels[0]
    if len(labels) <= 4:
        return "_".join(labels)
    return f"{labels[0]}_to_{labels[-1]}_{len(labels)}runs"


def default_output_path(input_paths: list[Path]) -> Path:
    return Path(f"applied_labr_ecal_model_{run_label_from_paths(input_paths)}.root")


def first_existing_tree(root_file: Any, preferred: str | None) -> str:
    if preferred:
        if preferred in root_file:
            return preferred
        if f"{preferred};1" in root_file:
            return f"{preferred};1"
        raise KeyError(f"Tree {preferred!r} was not found")

    for candidate in ("data", "T", "Tree", "tree"):
        if candidate in root_file:
            return candidate
        if f"{candidate};1" in root_file:
            return f"{candidate};1"

    tree_names = [
        key
        for key in root_file.keys()
        if getattr(root_file[key], "classname", "").startswith("TTree")
    ]
    if not tree_names:
        raise KeyError("No TTree objects found")
    return tree_names[0]


def entries_to_read(tree: Any, fraction: float | None, stop: int | None) -> int:
    total_entries = tree.num_entries
    selected = total_entries
    if fraction is not None:
        selected = max(1, int(total_entries * fraction))
    if stop is not None:
        selected = min(selected, stop)
    return selected


def collect_file_specs(
    input_paths: list[Path],
    tree_name_arg: str | None,
    fraction: float | None,
    stop: int | None,
) -> list[tuple[Path, str, int]]:
    file_specs: list[tuple[Path, str, int]] = []
    for input_path in input_paths:
        with uproot.open(input_path) as root_file:
            tree_name = first_existing_tree(root_file, tree_name_arg)
            tree = root_file[tree_name]
            if LABR_ENERGY_BRANCH not in tree.keys():
                raise KeyError(f"{input_path}:{tree_name} is missing {LABR_ENERGY_BRANCH}")
            file_specs.append((input_path, tree_name, entries_to_read(tree, fraction, stop)))
    return file_specs


def predict_chunk(model_bundle: dict[str, Any], energy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    energy = energy[:, :18]
    valid = np.isfinite(energy) & (energy > 0)
    trained_detectors = set(int(det) for det in model_bundle["detectors"])
    for det in range(18):
        if det not in trained_detectors:
            valid[:, det] = False

    event_indices, detector_indices = np.where(valid)
    raw_values = energy[valid].astype(float)
    features = make_feature_matrix(
        raw_values,
        detector_indices.astype(int),
        model_bundle["detectors"],
    )
    predicted = model_bundle["model"].predict(features)
    return event_indices, detector_indices, raw_values, predicted


def collect_predictions(
    file_specs: list[tuple[Path, str, int]],
    model_bundle: dict[str, Any],
    chunk_size: str,
) -> dict[str, np.ndarray]:
    file_indices = []
    entries = []
    detectors = []
    raw_values = []
    ecal_values = []

    total_entries = sum(entry_stop for _, _, entry_stop in file_specs)
    with tqdm(total=total_entries, unit="events", desc="Applying model") as pbar:
        for file_index, (input_path, tree_name, entry_stop) in enumerate(file_specs):
            entry_offset = 0
            with uproot.open(input_path) as root_file:
                tree = root_file[tree_name]
                for arrays in tree.iterate(
                    [LABR_ENERGY_BRANCH],
                    library="np",
                    entry_stop=entry_stop,
                    step_size=chunk_size,
                ):
                    energy = arrays[LABR_ENERGY_BRANCH]
                    event_indices, detector_indices, chunk_raw, chunk_ecal = predict_chunk(
                        model_bundle,
                        energy,
                    )
                    count = chunk_raw.size
                    if count:
                        file_indices.append(np.full(count, file_index, dtype=np.int32))
                        entries.append((event_indices + entry_offset).astype(np.int64))
                        detectors.append(detector_indices.astype(np.int32))
                        raw_values.append(chunk_raw.astype(np.float64))
                        ecal_values.append(chunk_ecal.astype(np.float64))
                    entry_offset += len(energy)
                    pbar.update(len(energy))

    if not raw_values:
        raise ValueError("No valid LaBr energies matched the trained detector set")

    return {
        "file_index": np.concatenate(file_indices),
        "entry": np.concatenate(entries),
        "detector": np.concatenate(detectors),
        "labr_energy": np.concatenate(raw_values),
        "labr_ecal_pred": np.concatenate(ecal_values),
    }


def write_file_map(file_specs: list[tuple[Path, str, int]], output_path: Path) -> Path:
    map_path = output_path.with_suffix(".files.txt")
    with open(map_path, "w") as handle:
        for file_index, (input_path, _, _) in enumerate(file_specs):
            handle.write(f"{file_index} {input_path}\n")
    return map_path


def write_root_output(
    root: Any,
    predictions: dict[str, np.ndarray],
    output_path: Path,
    raw_bin_width: float,
    ecal_bin_width: float,
    png_dir: Path | None,
) -> None:
    raw_values = predictions["labr_energy"]
    ecal_values = predictions["labr_ecal_pred"]
    detector_values = predictions["detector"]

    raw_bins, raw_low, raw_high = binned_range(raw_values, raw_bin_width)
    ecal_bins, ecal_low, ecal_high = binned_range(ecal_values, ecal_bin_width)
    raw_counts = np.histogram(raw_values, bins=raw_bins, range=(raw_low, raw_high))[0]
    ecal_counts = np.histogram(ecal_values, bins=ecal_bins, range=(ecal_low, ecal_high))[0]

    raw_hist = make_root_histogram(
        root,
        "labr_energy_raw",
        "Raw LaBr energy;LaBr energy;Counts",
        raw_counts,
        raw_low,
        raw_high,
    )
    ecal_hist = make_root_histogram(
        root,
        "labr_ecal_pred",
        "Predicted calibrated LaBr energy;Ecal;Counts",
        ecal_counts,
        ecal_low,
        ecal_high,
    )
    det_vs_eraw = root.TH2D(
        "detector_index_vs_labr_energy",
        "Detector index vs raw LaBr energy;Detector index;LaBr energy",
        18,
        -0.5,
        17.5,
        raw_bins,
        raw_low,
        raw_high,
    )
    det_vs_ecal = root.TH2D(
        "detector_index_vs_labr_ecal_pred",
        "Detector index vs predicted calibrated LaBr energy;Detector index;Ecal",
        18,
        -0.5,
        17.5,
        ecal_bins,
        ecal_low,
        ecal_high,
    )
    for detector, raw_energy, ecal in zip(detector_values, raw_values, ecal_values):
        det_vs_eraw.Fill(float(detector), float(raw_energy))
        det_vs_ecal.Fill(float(detector), float(ecal))

    with uproot.recreate(output_path) as out:
        out["LaBrEcalModel"] = predictions

    output_file = root.TFile(str(output_path), "UPDATE")
    raw_hist.Write()
    ecal_hist.Write()
    det_vs_eraw.Write()
    det_vs_ecal.Write()
    output_file.Close()

    if png_dir:
        png_dir.mkdir(parents=True, exist_ok=True)
        canvas = root.TCanvas("canvas", "canvas", 1000, 700)
        raw_hist.Draw()
        canvas.SaveAs(str(png_dir / "labr_energy_raw.png"))
        canvas.Clear()
        ecal_hist.Draw()
        canvas.SaveAs(str(png_dir / "labr_ecal_pred.png"))
        canvas.Clear()
        det_vs_eraw.Draw("COLZ")
        canvas.SaveAs(str(png_dir / "detector_index_vs_labr_energy.png"))
        canvas.Clear()
        det_vs_ecal.Draw("COLZ")
        canvas.SaveAs(str(png_dir / "detector_index_vs_labr_ecal_pred.png"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply trained Ei->Ecal model to raw LaBr energies in ROOT files."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help=(
            "One or more ROOT files or glob patterns. If the final argument is "
            "numeric, it is treated as percentage. Example: run*.root 1"
        ),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("ei_to_ecal_model.joblib"),
        help="Trained model from train_ei_ecal_model.py.",
    )
    parser.add_argument("--tree", help="Input tree name. Defaults to data/first TTree.")
    parser.add_argument("--stop", type=int, help="Maximum entries per file to read.")
    parser.add_argument("--chunk-size", default="100 MB", help="uproot chunk size.")
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Output ROOT file. Default includes the input run label, e.g. "
            "applied_labr_ecal_model_run-0989.root."
        ),
    )
    parser.add_argument("--raw-bin-width", type=float, default=10.0)
    parser.add_argument("--ecal-bin-width", type=float, default=1.0)
    parser.add_argument("--png-dir", type=Path, help="Optional PNG output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.raw_bin_width <= 0:
        raise ValueError("--raw-bin-width must be greater than 0")
    if args.ecal_bin_width <= 0:
        raise ValueError("--ecal-bin-width must be greater than 0")

    root = import_root()
    joblib = import_joblib()
    model_bundle = joblib.load(args.model)
    input_patterns, percentage = split_inputs_and_percentage(args.inputs)
    input_paths = expand_input_patterns(input_patterns)
    output_path = args.output or default_output_path(input_paths)
    fraction = percentage_to_fraction(percentage)
    file_specs = collect_file_specs(input_paths, args.tree, fraction, args.stop)

    predictions = collect_predictions(file_specs, model_bundle, args.chunk_size)
    write_root_output(
        root,
        predictions,
        output_path,
        args.raw_bin_width,
        args.ecal_bin_width,
        args.png_dir,
    )
    file_map = write_file_map(file_specs, output_path)

    print(f"Predicted rows: {len(predictions['labr_energy'])}")
    print(f"Detectors used: {', '.join(map(str, model_bundle['detectors']))}")
    print(f"Wrote {output_path}")
    print(f"Wrote {file_map}")


if __name__ == "__main__":
    main()
