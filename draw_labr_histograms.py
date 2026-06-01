# Author: Tawfik Gaballah
# GitHub: tawfikgaballah
# Project: ML-LaBr3-Calibration

"""Draw LaBr energy and calibrated-energy histograms from raw ROOT files."""

from __future__ import annotations

import argparse
import csv
import glob
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm
import uproot


LABR_ECAL_BRANCH = "rootout/labr/labr.ecal[18]"
LABR_ENERGY_BRANCH = "rootout/labr/labr.energy[18]"
DEFAULT_WORKERS = max(1, min(4, os.cpu_count() or 1))
DEFAULT_COEFFICIENTS = Path("labr_energy_calibration_coefficients.csv")
DEFAULT_ECAL_BIN_WIDTH = 1.0
DEFAULT_ENERGY_BIN_WIDTH = 10.0
DEFAULT_MAX_ECAL_BINS = 100000
DEFAULT_MAX_ENERGY_BINS = 100000


def import_root():
    try:
        import ROOT
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "This script needs PyROOT. On the analysis server, load the ROOT "
            "environment/module before running it."
        ) from exc
    return ROOT


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


def load_calibration_coefficients(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(
            f"Calibration coefficient file {path} was not found. Run "
            "calibrate_labr_energy_hists.py first, or pass --use-root-ecal "
            "to use the ROOT labr.ecal branch instead."
        )

    intercepts = np.zeros(18, dtype=float)
    slopes = np.zeros(18, dtype=float)
    has_coeff = np.zeros(18, dtype=bool)

    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            det = int(row["detector"])
            if det < 0 or det >= 18:
                continue
            intercepts[det] = float(row["intercept"])
            slopes[det] = float(row["slope"])
            has_coeff[det] = True

    if not np.any(has_coeff):
        raise ValueError(f"No detector coefficients were found in {path}")
    missing = [str(det) for det in range(18) if not has_coeff[det]]
    if missing:
        print(
            "Warning: missing calibration coefficients for detector(s): "
            + ", ".join(missing)
            + ". Those detectors will be skipped."
        )
    return intercepts, slopes, has_coeff


def padded_range_from_min_max(
    low: float | None,
    high: float | None,
    fallback=(-1.0, 1.0),
) -> tuple[float, float]:
    if low is None or high is None:
        return fallback
    if low == high:
        pad = max(abs(low) * 0.05, 1.0)
    else:
        pad = (high - low) * 0.05
    return low - pad, high + pad


def binned_range(
    hist_range: tuple[float, float],
    bin_width: float,
) -> tuple[tuple[float, float], int]:
    low, high = hist_range
    bins = max(1, int(np.ceil((high - low) / bin_width)))
    return (low, low + bins * bin_width), bins


def limit_range_bins(
    label: str,
    hist_range: tuple[float, float],
    bin_width: float,
    max_bins: int,
    manual_range: bool,
) -> tuple[float, float]:
    if manual_range:
        return hist_range
    bins = max(1, int(np.ceil((hist_range[1] - hist_range[0]) / bin_width)))
    if bins <= max_bins:
        return hist_range

    limited_low = 0.0 if hist_range[0] < 0 < hist_range[1] else hist_range[0]
    limited_high = limited_low + max_bins * bin_width
    print(
        f"Warning: {label} automatic range {hist_range[0]:.3f} to {hist_range[1]:.3f} "
        f"would create {bins} bins. Using {limited_low:.3f} to {limited_high:.3f} "
        f"({max_bins} bins). Pass --{label}-range LOW HIGH to override."
    )
    return limited_low, limited_high


def update_min_max(
    values: np.ndarray,
    current_low: float | None,
    current_high: float | None,
) -> tuple[float | None, float | None]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return current_low, current_high
    low = float(values.min())
    high = float(values.max())
    if current_low is None or low < current_low:
        current_low = low
    if current_high is None or high > current_high:
        current_high = high
    return current_low, current_high


def merge_min_max(
    current_low: float | None,
    current_high: float | None,
    other_low: float | None,
    other_high: float | None,
) -> tuple[float | None, float | None]:
    if other_low is None or other_high is None:
        return current_low, current_high
    if current_low is None or other_low < current_low:
        current_low = other_low
    if current_high is None or other_high > current_high:
        current_high = other_high
    return current_low, current_high


def entries_to_read(tree: Any, fraction: float | None, stop: int | None) -> int:
    total_entries = tree.num_entries
    selected = total_entries
    if fraction is not None:
        selected = max(1, int(total_entries * fraction))
    if stop is not None:
        selected = min(selected, stop)
    return selected


def make_work_specs(
    file_specs: list[tuple[Path, str, int]],
    entries_per_task: int,
) -> list[tuple[str, str, int, int]]:
    work_specs: list[tuple[str, str, int, int]] = []
    for input_path, tree_name, entry_stop in file_specs:
        for entry_start in range(0, entry_stop, entries_per_task):
            work_specs.append(
                (
                    str(input_path),
                    tree_name,
                    entry_start,
                    min(entry_start + entries_per_task, entry_stop),
                )
            )
    return work_specs


def labr_arrays_from_chunk(
    arrays: dict[str, np.ndarray],
    use_root_ecal: bool,
    intercepts: np.ndarray,
    slopes: np.ndarray,
    has_coeff: np.ndarray,
    include_invalid: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    energy = arrays[LABR_ENERGY_BRANCH][:, :18]
    if use_root_ecal:
        ecal = arrays[LABR_ECAL_BRANCH][:, :18]
    else:
        ecal = intercepts.reshape(1, -1) + slopes.reshape(1, -1) * energy

    detector_valid = np.broadcast_to(has_coeff.reshape(1, -1), energy.shape).copy()
    if not include_invalid:
        energy_valid = detector_valid & (energy > 0)
        ecal_valid = detector_valid & (ecal > 0)
    else:
        energy_valid = detector_valid
        ecal_valid = detector_valid

    return energy, ecal, energy_valid, ecal_valid


def scan_range_task(args):
    (
        input_path,
        tree_name,
        entry_start,
        entry_stop,
        chunk_size,
        include_invalid,
        use_root_ecal,
        intercepts,
        slopes,
        has_coeff,
    ) = args
    ecal_low = None
    ecal_high = None
    energy_low = None
    energy_high = None
    branches = [LABR_ENERGY_BRANCH]
    if use_root_ecal:
        branches.append(LABR_ECAL_BRANCH)

    with uproot.open(input_path) as root_file:
        tree = root_file[tree_name]
        for arrays in tree.iterate(
            branches,
            library="np",
            entry_start=entry_start,
            entry_stop=entry_stop,
            step_size=chunk_size,
        ):
            energy, ecal, energy_valid, ecal_valid = labr_arrays_from_chunk(
                arrays,
                use_root_ecal,
                intercepts,
                slopes,
                has_coeff,
                include_invalid,
            )
            ecal = ecal[ecal_valid]
            energy = energy[energy_valid]
            ecal_low, ecal_high = update_min_max(ecal, ecal_low, ecal_high)
            energy_low, energy_high = update_min_max(
                energy, energy_low, energy_high
            )

    return entry_stop - entry_start, ecal_low, ecal_high, energy_low, energy_high


def histogram_count_task(args):
    (
        input_path,
        tree_name,
        entry_start,
        entry_stop,
        chunk_size,
        include_invalid,
        ecal_bins,
        energy_bins,
        ecal_range,
        energy_range,
        use_root_ecal,
        intercepts,
        slopes,
        has_coeff,
    ) = args

    ecal_counts_all = np.zeros(ecal_bins, dtype=float)
    energy_counts_all = np.zeros(energy_bins, dtype=float)
    ecal_counts_det = np.zeros((18, ecal_bins), dtype=float)
    energy_counts_det = np.zeros((18, energy_bins), dtype=float)
    branches = [LABR_ENERGY_BRANCH]
    if use_root_ecal:
        branches.append(LABR_ECAL_BRANCH)

    with uproot.open(input_path) as root_file:
        tree = root_file[tree_name]
        for arrays in tree.iterate(
            branches,
            library="np",
            entry_start=entry_start,
            entry_stop=entry_stop,
            step_size=chunk_size,
        ):
            energy, ecal, energy_valid, ecal_valid = labr_arrays_from_chunk(
                arrays,
                use_root_ecal,
                intercepts,
                slopes,
                has_coeff,
                include_invalid,
            )

            ecal_counts_all += np.histogram(
                ecal[ecal_valid], bins=ecal_bins, range=ecal_range
            )[0]
            energy_counts_all += np.histogram(
                energy[energy_valid], bins=energy_bins, range=energy_range
            )[0]

            for det in np.flatnonzero(has_coeff):
                det_ecal_valid = ecal_valid[:, det]
                det_energy_valid = energy_valid[:, det]
                ecal_counts_det[det] += np.histogram(
                    ecal[:, det][det_ecal_valid], bins=ecal_bins, range=ecal_range
                )[0]
                energy_counts_det[det] += np.histogram(
                    energy[:, det][det_energy_valid], bins=energy_bins, range=energy_range
                )[0]

    return (
        entry_stop - entry_start,
        ecal_counts_all,
        energy_counts_all,
        ecal_counts_det,
        energy_counts_det,
    )


def collect_ranges(
    input_paths: list[Path],
    tree_name_arg: str | None,
    fraction: float | None,
    stop: int | None,
    chunk_size: str,
    include_invalid: bool,
    workers: int,
    entries_per_task: int,
    use_root_ecal: bool,
    intercepts: np.ndarray,
    slopes: np.ndarray,
    has_coeff: np.ndarray,
) -> tuple[tuple[float, float], tuple[float, float], list[tuple[Path, str, int]]]:
    ecal_low: float | None = None
    ecal_high: float | None = None
    energy_low: float | None = None
    energy_high: float | None = None
    file_specs = collect_file_specs(input_paths, tree_name_arg, fraction, stop, use_root_ecal)
    work_specs = make_work_specs(file_specs, entries_per_task)
    total_range_entries = sum(entry_stop - entry_start for _, _, entry_start, entry_stop in work_specs)

    task_args = [
        (
            input_path,
            tree_name,
            entry_start,
            entry_stop,
            chunk_size,
            include_invalid,
            use_root_ecal,
            intercepts,
            slopes,
            has_coeff,
        )
        for input_path, tree_name, entry_start, entry_stop in work_specs
    ]

    with tqdm(total=total_range_entries, unit="events", desc="Scanning ranges") as pbar:
        if workers == 1:
            for task_arg in task_args:
                completed, task_ecal_low, task_ecal_high, task_energy_low, task_energy_high = scan_range_task(task_arg)
                pbar.update(completed)
                ecal_low, ecal_high = merge_min_max(
                    ecal_low, ecal_high, task_ecal_low, task_ecal_high
                )
                energy_low, energy_high = merge_min_max(
                    energy_low, energy_high, task_energy_low, task_energy_high
                )
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(scan_range_task, task_arg) for task_arg in task_args]
                for future in as_completed(futures):
                    completed, task_ecal_low, task_ecal_high, task_energy_low, task_energy_high = future.result()
                    pbar.update(completed)
                    ecal_low, ecal_high = merge_min_max(
                        ecal_low, ecal_high, task_ecal_low, task_ecal_high
                    )
                    energy_low, energy_high = merge_min_max(
                        energy_low, energy_high, task_energy_low, task_energy_high
                    )

    return (
        padded_range_from_min_max(ecal_low, ecal_high),
        padded_range_from_min_max(energy_low, energy_high),
        file_specs,
    )


def collect_file_specs(
    input_paths: list[Path],
    tree_name_arg: str | None,
    fraction: float | None,
    stop: int | None,
    use_root_ecal: bool,
) -> list[tuple[Path, str, int]]:
    file_specs: list[tuple[Path, str, int]] = []
    required = [LABR_ENERGY_BRANCH]
    if use_root_ecal:
        required.append(LABR_ECAL_BRANCH)
    for input_path in input_paths:
        with uproot.open(input_path) as root_file:
            tree_name = first_existing_tree(root_file, tree_name_arg)
            tree = root_file[tree_name]
            missing = [
                name
                for name in required
                if name not in tree.keys()
            ]
            if missing:
                raise KeyError(f"{input_path}:{tree_name} is missing {missing}")
            file_specs.append((input_path, tree_name, entries_to_read(tree, fraction, stop)))
    return file_specs


def fill_histograms(
    root: Any,
    file_specs: list[tuple[Path, str, int]],
    ecal_range: tuple[float, float],
    energy_range: tuple[float, float],
    ecal_bins: int,
    energy_bins: int,
    chunk_size: str,
    include_invalid: bool,
    workers: int,
    entries_per_task: int,
    use_root_ecal: bool,
    intercepts: np.ndarray,
    slopes: np.ndarray,
    has_coeff: np.ndarray,
) -> dict[str, Any]:
    ecal_counts_all = np.zeros(ecal_bins, dtype=float)
    energy_counts_all = np.zeros(energy_bins, dtype=float)
    ecal_counts_det = np.zeros((18, ecal_bins), dtype=float)
    energy_counts_det = np.zeros((18, energy_bins), dtype=float)

    work_specs = make_work_specs(file_specs, entries_per_task)
    total_entries = sum(entry_stop - entry_start for _, _, entry_start, entry_stop in work_specs)
    task_args = [
        (
            input_path,
            tree_name,
            entry_start,
            entry_stop,
            chunk_size,
            include_invalid,
            ecal_bins,
            energy_bins,
            ecal_range,
            energy_range,
            use_root_ecal,
            intercepts,
            slopes,
            has_coeff,
        )
        for input_path, tree_name, entry_start, entry_stop in work_specs
    ]

    with tqdm(total=total_entries, unit="events", desc="Filling histograms") as pbar:
        if workers == 1:
            for task_arg in task_args:
                (
                    completed,
                    task_ecal_all,
                    task_energy_all,
                    task_ecal_det,
                    task_energy_det,
                ) = histogram_count_task(task_arg)
                pbar.update(completed)
                ecal_counts_all += task_ecal_all
                energy_counts_all += task_energy_all
                ecal_counts_det += task_ecal_det
                energy_counts_det += task_energy_det
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(histogram_count_task, task_arg)
                    for task_arg in task_args
                ]
                for future in as_completed(futures):
                    (
                        completed,
                        task_ecal_all,
                        task_energy_all,
                        task_ecal_det,
                        task_energy_det,
                    ) = future.result()
                    pbar.update(completed)
                    ecal_counts_all += task_ecal_all
                    energy_counts_all += task_energy_all
                    ecal_counts_det += task_ecal_det
                    energy_counts_det += task_energy_det

    print_detector_count_summary(energy_counts_det, ecal_counts_det, has_coeff)

    histograms: dict[str, Any] = {
        "labr_ecal_all": make_root_histogram(
            root,
            "labr_ecal_all",
            "LaBr calibrated energy;LaBr ecal;Counts",
            ecal_counts_all,
            ecal_range,
        ),
        "labr_energy_all": make_root_histogram(
            root,
            "labr_energy_all",
            "LaBr raw energy;LaBr energy;Counts",
            energy_counts_all,
            energy_range,
        ),
        "labr_ecal_vs_detector_index": make_root_detector_index_histogram(
            root,
            "labr_ecal_vs_detector_index",
            "LaBr calibrated energy vs detector index;LaBr ecal;Detector index",
            ecal_counts_det,
            ecal_range,
        ),
        "labr_energy_vs_detector_index": make_root_detector_index_histogram(
            root,
            "labr_energy_vs_detector_index",
            "LaBr raw energy vs detector index;LaBr energy;Detector index",
            energy_counts_det,
            energy_range,
        ),
    }

    for det in np.flatnonzero(has_coeff):
        histograms[f"labr_ecal_det{det}"] = make_root_histogram(
            root,
            f"labr_ecal_det{det}",
            f"LaBr calibrated energy detector {det};LaBr ecal;Counts",
            ecal_counts_det[det],
            ecal_range,
        )
        histograms[f"labr_energy_det{det}"] = make_root_histogram(
            root,
            f"labr_energy_det{det}",
            f"LaBr raw energy detector {det};LaBr energy;Counts",
            energy_counts_det[det],
            energy_range,
        )

    return histograms


def print_detector_count_summary(
    energy_counts_det: np.ndarray,
    ecal_counts_det: np.ndarray,
    has_coeff: np.ndarray,
) -> None:
    print("Detector histogram entries kept:")
    for det in range(18):
        if not has_coeff[det]:
            continue
        energy_entries = int(np.sum(energy_counts_det[det]))
        ecal_entries = int(np.sum(ecal_counts_det[det]))
        print(f"  det {det:2d}: energy={energy_entries:8d}  ecal={ecal_entries:8d}")


def make_root_histogram(
    root: Any,
    name: str,
    title: str,
    counts: np.ndarray,
    hist_range: tuple[float, float],
) -> Any:
    histogram = root.TH1D(name, title, len(counts), hist_range[0], hist_range[1])
    for index, count in enumerate(counts, start=1):
        histogram.SetBinContent(index, float(count))
    return histogram


def make_root_detector_index_histogram(
    root: Any,
    name: str,
    title: str,
    counts_by_detector: np.ndarray,
    energy_range: tuple[float, float],
) -> Any:
    x_bins = counts_by_detector.shape[1]
    histogram = root.TH2D(
        name,
        title,
        x_bins,
        energy_range[0],
        energy_range[1],
        18,
        0,
        18,
    )
    for det in range(18):
        for x_index, count in enumerate(counts_by_detector[det], start=1):
            histogram.SetBinContent(x_index, det + 1, float(count))
    return histogram


def save_pngs(root: Any, histograms: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    canvas = root.TCanvas("canvas", "canvas", 1000, 700)
    for name, histogram in histograms.items():
        canvas.Clear()
        if histogram.InheritsFrom("TH2"):
            histogram.Draw("COLZ")
        else:
            histogram.Draw()
        canvas.SaveAs(str(output_dir / f"{name}.png"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw LaBr ecal and raw-energy histograms from ROOT files."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help=(
            "One or more ROOT files or glob patterns. If the final argument is "
            "numeric, it is treated as percentage. Example: run*.root 1"
        ),
    )
    parser.add_argument("--tree", help="Input tree name. Defaults to data/first TTree.")
    parser.add_argument("--stop", type=int, help="Maximum entries per file to read.")
    parser.add_argument(
        "--coefficients",
        type=Path,
        default=DEFAULT_COEFFICIENTS,
        help=(
            "Linear calibration coefficients CSV from calibrate_labr_energy_hists.py. "
            f"Default: {DEFAULT_COEFFICIENTS}."
        ),
    )
    parser.add_argument(
        "--use-root-ecal",
        action="store_true",
        help="Use the ROOT labr.ecal branch instead of calculating ecal from coefficients.",
    )
    parser.add_argument(
        "--energy-bin-width",
        type=float,
        default=DEFAULT_ENERGY_BIN_WIDTH,
        help=f"Raw labr.energy histogram bin width. Default: {DEFAULT_ENERGY_BIN_WIDTH}.",
    )
    parser.add_argument(
        "--ecal-bin-width",
        type=float,
        default=DEFAULT_ECAL_BIN_WIDTH,
        help=f"Calculated labr.ecal histogram bin width. Default: {DEFAULT_ECAL_BIN_WIDTH}.",
    )
    parser.add_argument("--chunk-size", default="100 MB", help="uproot chunk size.")
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Parallel worker processes. Default: {DEFAULT_WORKERS}. Use 1 to disable.",
    )
    parser.add_argument(
        "--entries-per-task",
        type=int,
        default=200000,
        help="Entries assigned to each parallel task. Default: 200000.",
    )
    parser.add_argument(
        "--ecal-range",
        nargs=2,
        type=float,
        metavar=("LOW", "HIGH"),
        help="Manual LaBr ecal histogram range. Skips ecal range scanning.",
    )
    parser.add_argument(
        "--energy-range",
        nargs=2,
        type=float,
        metavar=("LOW", "HIGH"),
        help="Manual LaBr energy histogram range. Skips energy range scanning.",
    )
    parser.add_argument(
        "--max-ecal-bins",
        type=int,
        default=DEFAULT_MAX_ECAL_BINS,
        help=(
            "Maximum ecal bins for automatic ranges. "
            f"Default: {DEFAULT_MAX_ECAL_BINS}."
        ),
    )
    parser.add_argument(
        "--max-energy-bins",
        type=int,
        default=DEFAULT_MAX_ENERGY_BINS,
        help=(
            "Maximum raw-energy bins for automatic ranges. "
            f"Default: {DEFAULT_MAX_ENERGY_BINS}."
        ),
    )
    parser.add_argument(
        "--include-invalid",
        action="store_true",
        help="Include zero/negative placeholder values. Default keeps only values > 0.",
    )
    parser.add_argument(
        "--output",
        default="labr_histograms.root",
        help="Output ROOT file. Default: labr_histograms.root.",
    )
    parser.add_argument("--png-dir", type=Path, help="Optional directory for PNG plots.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.entries_per_task < 1:
        raise ValueError("--entries-per-task must be at least 1")
    if args.energy_bin_width <= 0:
        raise ValueError("--energy-bin-width must be greater than 0")
    if args.ecal_bin_width <= 0:
        raise ValueError("--ecal-bin-width must be greater than 0")
    if args.max_ecal_bins <= 0:
        raise ValueError("--max-ecal-bins must be greater than 0")
    if args.max_energy_bins <= 0:
        raise ValueError("--max-energy-bins must be greater than 0")
    root = import_root()
    input_patterns, percentage = split_inputs_and_percentage(args.inputs)
    input_paths = expand_input_patterns(input_patterns)
    fraction = percentage_to_fraction(percentage)
    use_root_ecal = args.use_root_ecal
    if use_root_ecal:
        intercepts = np.zeros(18, dtype=float)
        slopes = np.ones(18, dtype=float)
        has_coeff = np.ones(18, dtype=bool)
        print("Using ROOT labr.ecal branch for all detectors.")
    else:
        intercepts, slopes, has_coeff = load_calibration_coefficients(args.coefficients)
        print(f"Calculating labr_ecal from coefficients in {args.coefficients}")

    if args.ecal_range and args.energy_range:
        ecal_range = tuple(args.ecal_range)
        energy_range = tuple(args.energy_range)
        file_specs = collect_file_specs(
            input_paths,
            args.tree,
            fraction,
            args.stop,
            use_root_ecal,
        )
    else:
        ecal_range, energy_range, file_specs = collect_ranges(
            input_paths,
            args.tree,
            fraction,
            args.stop,
            args.chunk_size,
            args.include_invalid,
            args.workers,
            args.entries_per_task,
            use_root_ecal,
            intercepts,
            slopes,
            has_coeff,
        )
        if args.ecal_range:
            ecal_range = tuple(args.ecal_range)
        if args.energy_range:
            energy_range = tuple(args.energy_range)
    ecal_range = limit_range_bins(
        "ecal",
        ecal_range,
        args.ecal_bin_width,
        args.max_ecal_bins,
        args.ecal_range is not None,
    )
    energy_range = limit_range_bins(
        "energy",
        energy_range,
        args.energy_bin_width,
        args.max_energy_bins,
        args.energy_range is not None,
    )
    ecal_range, ecal_bins = binned_range(ecal_range, args.ecal_bin_width)
    energy_range, energy_bins = binned_range(energy_range, args.energy_bin_width)
    print(f"LaBr ecal range: {ecal_range[0]:.3f} to {ecal_range[1]:.3f}")
    print(f"LaBr energy range: {energy_range[0]:.3f} to {energy_range[1]:.3f}")
    print(f"LaBr ecal bins: {ecal_bins} with width {args.ecal_bin_width}")
    print(f"LaBr energy bins: {energy_bins} with width {args.energy_bin_width}")

    histograms = fill_histograms(
        root,
        file_specs,
        ecal_range,
        energy_range,
        ecal_bins,
        energy_bins,
        args.chunk_size,
        args.include_invalid,
        args.workers,
        args.entries_per_task,
        use_root_ecal,
        intercepts,
        slopes,
        has_coeff,
    )

    output_file = root.TFile(args.output, "RECREATE")
    for histogram in histograms.values():
        histogram.Write()
    output_file.Close()

    if args.png_dir:
        save_pngs(root, histograms, args.png_dir)

    print(f"Wrote {args.output}")
    if args.png_dir:
        print(f"Wrote PNG files to {args.png_dir}")


if __name__ == "__main__":
    main()
