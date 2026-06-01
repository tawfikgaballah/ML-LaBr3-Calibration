"""Apply LaBr energy calibration and timing-model corrections to new ROOT runs."""

from __future__ import annotations

import argparse
import csv
import glob
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import uproot
from tqdm import tqdm

from offset_aware_timing_model import OffsetAwareTimingModel

try:
    from sklearn.base import BaseEstimator, TransformerMixin
except ModuleNotFoundError:
    class BaseEstimator:
        pass

    class TransformerMixin:
        pass


LABR_ENERGY_BRANCH = "rootout/labr/labr.energy[18]"
LABR_TIME_BRANCH = "rootout/labr/labr.time[18]"
DYTIME_BRANCH = "rootout/NpspmtCeBr/NpspmtCeBr.dytime"
FEATURE_COLUMNS = ["Ei", "index_i", "Ej", "index_j"]
DEFAULT_COEFFICIENTS = Path("labr_energy_calibration_coefficients.csv")
DEFAULT_MODEL = Path("trained_model.joblib")


class Log1pTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.log1p(X)


class Expm1Transformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.expm1(X)


def load_timing_model(path: Path):
    try:
        import joblib
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "This script needs joblib and scikit-learn to load the timing model. "
            "Activate the ML virtual environment and install requirements first."
        ) from exc
    model = joblib.load(path)
    if hasattr(model, "predict_time_walk"):
        return model
    return OffsetAwareTimingModel(model)


def predict_tdiff(model: Any, features: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_time_walk"):
        return np.asarray(model.predict_time_walk(features), dtype=float).reshape(-1)
    try:
        predictions = model.predict(features, regressor__verbose=0)
    except TypeError:
        try:
            predictions = model.predict(features, verbose=0)
        except TypeError:
            predictions = model.predict(features)
    return np.asarray(predictions, dtype=float).reshape(-1)


def detector_pair_offsets(index_i: np.ndarray, index_j: np.ndarray, detector_offsets: np.ndarray) -> np.ndarray:
    return detector_offsets[index_i] - detector_offsets[index_j]


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
    return Path(f"time_corrected_{run_label_from_paths(input_paths)}.root")


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


def scalar(value: Any) -> float:
    array = np.asarray(value)
    if array.ndim == 0:
        return array.item()
    if array.size == 0:
        return np.nan
    return array.reshape(-1)[0].item()


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
    require_dynode: bool,
) -> list[tuple[Path, str, int]]:
    branches = [LABR_ENERGY_BRANCH, LABR_TIME_BRANCH]
    if require_dynode:
        branches.append(DYTIME_BRANCH)
    file_specs: list[tuple[Path, str, int]] = []
    for input_path in input_paths:
        with uproot.open(input_path) as root_file:
            tree_name = first_existing_tree(root_file, tree_name_arg)
            tree = root_file[tree_name]
            missing = [name for name in branches if name not in tree.keys()]
            if missing:
                raise KeyError(f"{input_path}:{tree_name} is missing {missing}")
            file_specs.append((input_path, tree_name, entries_to_read(tree, fraction, stop)))
    return file_specs


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
            if 0 <= detector < 18:
                intercepts[detector] = float(row["intercept"])
                slopes[detector] = float(row["slope"])
                has_coeff[detector] = True

    if not np.any(has_coeff):
        raise ValueError(f"No detector coefficients were found in {path}")
    return intercepts, slopes, has_coeff


def allowed_model_detectors(model: Any) -> tuple[set[int] | None, set[int] | None]:
    try:
        preprocessor = model.named_steps["preprocessor"]
        onehot = preprocessor.named_transformers_["onehot"]
        allowed_i = set(int(value) for value in onehot.categories_[0])
        allowed_j = set(int(value) for value in onehot.categories_[1])
        return allowed_i, allowed_j
    except Exception:
        return None, None


def build_pair_dataframe(
    arrays: dict[str, np.ndarray],
    intercepts: np.ndarray,
    slopes: np.ndarray,
    has_coeff: np.ndarray,
    min_ecal: float,
    min_time: float,
    require_dynode: bool,
    min_dynode_time: float,
    allowed_i: set[int] | None,
    allowed_j: set[int] | None,
) -> pd.DataFrame:
    rows: list[tuple[int, float, int, float, int, float]] = []
    raw_energy_array = arrays[LABR_ENERGY_BRANCH]
    time_array = arrays[LABR_TIME_BRANCH]
    dytime_array = arrays.get(DYTIME_BRANCH)

    for local_entry in range(len(raw_energy_array)):
        if require_dynode:
            dytime = scalar(dytime_array[local_entry])
            if not np.isfinite(dytime) or dytime <= min_dynode_time:
                continue

        raw_energy = np.asarray(raw_energy_array[local_entry], dtype=float)
        time_values = np.asarray(time_array[local_entry], dtype=float)
        detector_count = min(18, len(raw_energy), len(time_values), len(has_coeff))
        if detector_count == 0:
            continue

        ecal = intercepts[:detector_count] + slopes[:detector_count] * raw_energy[:detector_count]
        valid = (
            has_coeff[:detector_count]
            & np.isfinite(raw_energy[:detector_count])
            & np.isfinite(ecal)
            & np.isfinite(time_values[:detector_count])
            & (raw_energy[:detector_count] > 0)
            & (ecal > min_ecal)
            & (time_values[:detector_count] > min_time)
        )
        valid_indices = np.flatnonzero(valid)
        for index_i in valid_indices:
            if allowed_i is not None and index_i not in allowed_i:
                continue
            for index_j in valid_indices:
                if allowed_j is not None and index_j not in allowed_j:
                    continue
                rows.append(
                    (
                        local_entry,
                        float(ecal[index_i]),
                        int(index_i),
                        float(ecal[index_j]),
                        int(index_j),
                        float(time_values[index_i] - time_values[index_j]),
                    )
                )

    return pd.DataFrame(
        rows,
        columns=["local_entry", "Ei", "index_i", "Ej", "index_j", "T_Diff"],
    )


def iter_pair_chunks(
    file_specs: list[tuple[Path, str, int]],
    chunk_size: str,
    require_dynode: bool,
    pair_builder_args: tuple[Any, ...],
):
    branches = [LABR_ENERGY_BRANCH, LABR_TIME_BRANCH]
    if require_dynode:
        branches.append(DYTIME_BRANCH)
    for file_index, (input_path, tree_name, entry_stop) in enumerate(file_specs):
        entry_offset = 0
        with uproot.open(input_path) as root_file:
            tree = root_file[tree_name]
            for arrays in tree.iterate(
                branches,
                library="np",
                entry_stop=entry_stop,
                step_size=chunk_size,
            ):
                pair_df = build_pair_dataframe(arrays, *pair_builder_args)
                if not pair_df.empty:
                    pair_df.insert(0, "file_index", file_index)
                    pair_df["entry"] = pair_df["local_entry"] + entry_offset
                    pair_df = pair_df.drop(columns=["local_entry"])
                yield len(arrays[LABR_ENERGY_BRANCH]), pair_df
                entry_offset += len(arrays[LABR_ENERGY_BRANCH])


def padded_range_from_min_max(low: float | None, high: float | None, fallback=(-1.0, 1.0)):
    if low is None or high is None:
        return fallback
    if low == high:
        pad = max(abs(low) * 0.05, 1.0)
    else:
        pad = (high - low) * 0.05
    return low - pad, high + pad


def update_min_max(values: np.ndarray, low: float | None, high: float | None):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return low, high
    value_low = float(values.min())
    value_high = float(values.max())
    if low is None or value_low < low:
        low = value_low
    if high is None or value_high > high:
        high = value_high
    return low, high


def binned_axis(low: float, high: float, bin_width: float):
    bins = max(1, int(np.ceil((high - low) / bin_width)))
    return bins, low, low + bins * bin_width


def validate_range(name: str, values: list[float] | None) -> tuple[float, float] | None:
    if values is None:
        return None
    low, high = float(values[0]), float(values[1])
    if low >= high:
        raise ValueError(f"{name} low value must be smaller than high value")
    return low, high


def bin_count(value_range: tuple[float, float], bin_width: float) -> int:
    return max(1, int(np.ceil((value_range[1] - value_range[0]) / bin_width)))


def limit_time_range(
    label: str,
    value_range: tuple[float, float],
    bin_width: float,
    max_bins: int,
) -> tuple[float, float]:
    bins = bin_count(value_range, bin_width)
    if bins <= max_bins:
        return value_range

    low, high = value_range
    center = 0.0 if low <= 0.0 <= high else 0.5 * (low + high)
    half_width = 0.5 * max_bins * bin_width
    limited = (center - half_width, center + half_width)
    print(
        f"{label} histogram range {low:.6g} to {high:.6g} would need {bins} bins. "
        f"Using {limited[0]:.6g} to {limited[1]:.6g} ({max_bins} bins). "
        "Use --time-range or --corrected-time-range to override.",
        flush=True,
    )
    return limited


def limit_energy_range(
    value_range: tuple[float, float],
    bin_width: float,
    max_bins: int,
) -> tuple[float, float]:
    bins = bin_count(value_range, bin_width)
    if bins <= max_bins:
        return value_range

    low, high = value_range
    limited_low = max(0.0, low)
    limited_high = limited_low + max_bins * bin_width
    print(
        f"Energy histogram range {low:.6g} to {high:.6g} would need {bins} bins. "
        f"Using {limited_low:.6g} to {limited_high:.6g} ({max_bins} bins). "
        "Use --energy-range to override.",
        flush=True,
    )
    return limited_low, limited_high


def apply_histogram_range_limits(
    tdiff_range: tuple[float, float],
    corrected_range: tuple[float, float],
    ej_range: tuple[float, float],
    args: argparse.Namespace,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    manual_time_range = validate_range("--time-range", args.time_range)
    manual_corrected_range = validate_range("--corrected-time-range", args.corrected_time_range)
    manual_energy_range = validate_range("--energy-range", args.energy_range)

    if manual_time_range is not None:
        tdiff_range = manual_time_range
    if manual_corrected_range is not None:
        corrected_range = manual_corrected_range
    if manual_energy_range is not None:
        ej_range = manual_energy_range

    if manual_energy_range is None:
        ej_range = limit_energy_range(ej_range, args.energy_bin_width, args.max_energy_bins)

    energy_bins = bin_count(ej_range, args.energy_bin_width)
    max_time_bins_from_2d = max(1, args.max_2d_bins // max(1, energy_bins))
    max_time_bins = min(args.max_time_bins, max_time_bins_from_2d)

    if manual_time_range is None:
        tdiff_range = limit_time_range("T_Diff", tdiff_range, args.time_bin_width, max_time_bins)
    if manual_corrected_range is None:
        corrected_range = limit_time_range(
            "T_Diff_Corrected",
            corrected_range,
            args.time_bin_width,
            max_time_bins,
        )

    return tdiff_range, corrected_range, ej_range


def estimate_detector_offsets(
    file_specs,
    model,
    chunk_size,
    total_entries,
    require_dynode,
    pair_builder_args,
    args,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if args.no_offset_correction:
        print("Detector-offset correction disabled.", flush=True)
        return np.zeros(18, dtype=float), np.zeros((18, 18), dtype=float), np.zeros((18, 18), dtype=float)

    offset_range = validate_range("--offset-range", args.offset_range)
    if offset_range is None:
        raise ValueError("--offset-range is required when offset correction is enabled")

    if hasattr(model, "empty_offset_statistics"):
        pair_sum, pair_counts = model.empty_offset_statistics()
    else:
        pair_sum = np.zeros((18, 18), dtype=float)
        pair_counts = np.zeros((18, 18), dtype=float)
    with tqdm(total=total_entries, unit="events", desc="Estimating detector offsets") as pbar:
        for completed, pair_df in iter_pair_chunks(
            file_specs,
            chunk_size,
            require_dynode,
            pair_builder_args,
        ):
            pbar.update(completed)
            if pair_df.empty:
                continue
            if hasattr(model, "accumulate_offset_statistics"):
                model.accumulate_offset_statistics(
                    pair_df,
                    pair_sum,
                    pair_counts,
                    target_column="T_Diff",
                    offset_range=offset_range,
                )
            else:
                predictions = predict_tdiff(model, pair_df[FEATURE_COLUMNS])
                residual = pair_df["T_Diff"].to_numpy(dtype=float) - predictions
                index_i = pair_df["index_i"].to_numpy(dtype=np.intp)
                index_j = pair_df["index_j"].to_numpy(dtype=np.intp)
                mask = (
                    (index_i != index_j)
                    & np.isfinite(residual)
                    & (offset_range[0] <= residual)
                    & (residual <= offset_range[1])
                )
                if not np.any(mask):
                    continue
                np.add.at(pair_sum, (index_i[mask], index_j[mask]), residual[mask])
                np.add.at(pair_counts, (index_i[mask], index_j[mask]), 1.0)

    if hasattr(model, "solve_detector_offsets"):
        offsets = model.solve_detector_offsets(pair_sum, pair_counts, args.min_offset_pair_count)
    else:
        offsets = solve_detector_offsets(pair_sum, pair_counts, args.min_offset_pair_count)
    pair_means = np.divide(
        pair_sum,
        pair_counts,
        out=np.zeros_like(pair_sum),
        where=pair_counts > 0,
    )
    print_detector_offsets(offsets, pair_counts, args.min_offset_pair_count)
    return offsets, pair_counts, pair_means


def solve_detector_offsets(pair_sum: np.ndarray, pair_counts: np.ndarray, min_pair_count: int) -> np.ndarray:
    rows: list[np.ndarray] = []
    targets: list[float] = []
    weights: list[float] = []
    for index_i in range(18):
        for index_j in range(18):
            count = pair_counts[index_i, index_j]
            if index_i == index_j or count < min_pair_count:
                continue
            row = np.zeros(18, dtype=float)
            row[index_i] = 1.0
            row[index_j] = -1.0
            rows.append(row)
            targets.append(pair_sum[index_i, index_j] / count)
            weights.append(np.sqrt(count))

    if not rows:
        print(
            "No detector pairs had enough statistics for offset solving; using zero offsets.",
            flush=True,
        )
        return np.zeros(18, dtype=float)

    matrix = np.vstack(rows)
    target = np.asarray(targets, dtype=float)
    weight = np.asarray(weights, dtype=float)
    weighted_matrix = matrix * weight[:, np.newaxis]
    weighted_target = target * weight
    offsets, *_ = np.linalg.lstsq(weighted_matrix, weighted_target, rcond=None)
    active = np.flatnonzero(np.sum(pair_counts >= min_pair_count, axis=0) + np.sum(pair_counts >= min_pair_count, axis=1))
    if active.size:
        offsets[active] -= float(np.mean(offsets[active]))
    return offsets


def print_detector_offsets(offsets: np.ndarray, pair_counts: np.ndarray, min_pair_count: int) -> None:
    print("Estimated detector time offsets from this run:", flush=True)
    for detector, offset in enumerate(offsets):
        used_pairs = int(
            np.sum(pair_counts[detector, :] >= min_pair_count)
            + np.sum(pair_counts[:, detector] >= min_pair_count)
        )
        if used_pairs:
            print(f"  detector {detector:2d}: {offset: .6f} using {used_pairs} pair directions", flush=True)


def scan_ranges(file_specs, model, chunk_size, total_entries, require_dynode, pair_builder_args, detector_offsets):
    tdiff_low = tdiff_high = None
    corr_low = corr_high = None
    ej_low = ej_high = None
    with tqdm(total=total_entries, unit="events", desc="Scanning correction ranges") as pbar:
        for completed, pair_df in iter_pair_chunks(
            file_specs,
            chunk_size,
            require_dynode,
            pair_builder_args,
        ):
            pbar.update(completed)
            if pair_df.empty:
                continue
            predictions = predict_tdiff(model, pair_df[FEATURE_COLUMNS])
            index_i = pair_df["index_i"].to_numpy(dtype=np.intp)
            index_j = pair_df["index_j"].to_numpy(dtype=np.intp)
            if hasattr(model, "pair_offset_correction"):
                offset_correction = model.pair_offset_correction(index_i, index_j, detector_offsets)
            else:
                offset_correction = detector_pair_offsets(index_i, index_j, detector_offsets)
            corrected = pair_df["T_Diff"].to_numpy(dtype=float) - predictions - offset_correction
            tdiff_low, tdiff_high = update_min_max(pair_df["T_Diff"].to_numpy(), tdiff_low, tdiff_high)
            corr_low, corr_high = update_min_max(corrected, corr_low, corr_high)
            ej_low, ej_high = update_min_max(pair_df["Ej"].to_numpy(), ej_low, ej_high)
    return (
        padded_range_from_min_max(tdiff_low, tdiff_high),
        padded_range_from_min_max(corr_low, corr_high),
        padded_range_from_min_max(ej_low, ej_high, fallback=(0.0, 1.0)),
    )


def create_histogram_state(tdiff_range, corrected_range, ej_range, args) -> dict[str, Any]:
    time_bins, tdiff_min, tdiff_max = binned_axis(tdiff_range[0], tdiff_range[1], args.time_bin_width)
    corr_bins, corr_min, corr_max = binned_axis(corrected_range[0], corrected_range[1], args.time_bin_width)
    energy_bins, ej_min, ej_max = binned_axis(ej_range[0], ej_range[1], args.energy_bin_width)

    time_edges = np.linspace(tdiff_min, tdiff_max, time_bins + 1)
    corrected_edges = np.linspace(corr_min, corr_max, corr_bins + 1)
    energy_edges = np.linspace(ej_min, ej_max, energy_bins + 1)
    detector_edges = np.linspace(-0.5, 17.5, 19)

    state: dict[str, Any] = {
        "edges": {
            "time": time_edges,
            "corrected": corrected_edges,
            "energy": energy_edges,
            "detector": detector_edges,
        },
        "hist1d": {
            "Prompt_Response_Corrected": np.zeros(corr_bins, dtype=float),
            "Prompt_Response": np.zeros(time_bins, dtype=float),
        },
        "hist2d": {
            "Ej_Vs_T_Diff_Corrected": np.zeros((corr_bins, energy_bins), dtype=float),
            "Ej_Vs_T_Diff": np.zeros((time_bins, energy_bins), dtype=float),
        },
        "pair_counts": np.zeros((18, 18), dtype=float),
        "pair_sum_corrected": np.zeros((18, 18), dtype=float),
    }
    for det in range(18):
        if det == args.reference_detector:
            continue
        state["hist1d"][f"Labr{args.reference_detector}_minus_Labr{det}_Before"] = np.zeros(
            time_bins,
            dtype=float,
        )
        state["hist1d"][f"Labr{args.reference_detector}_minus_Labr{det}_After"] = np.zeros(
            corr_bins,
            dtype=float,
        )
    return state


def add_hist1d(state: dict[str, Any], name: str, values: np.ndarray, edge_name: str) -> None:
    counts, _ = np.histogram(values, bins=state["edges"][edge_name])
    state["hist1d"][name] += counts


def add_hist2d(
    state: dict[str, Any],
    name: str,
    x_values: np.ndarray,
    y_values: np.ndarray,
    x_edge_name: str,
    y_edge_name: str,
) -> None:
    counts, _, _ = np.histogram2d(
        x_values,
        y_values,
        bins=(state["edges"][x_edge_name], state["edges"][y_edge_name]),
    )
    state["hist2d"][name] += counts


def write_histograms(output_file: Any, state: dict[str, Any]) -> None:
    edges = state["edges"]
    for name, values in state["hist1d"].items():
        edge_name = "corrected" if name.endswith("_Corrected") or name.endswith("_After") else "time"
        output_file[name] = values, edges[edge_name]

    output_file["Ej_Vs_T_Diff_Corrected"] = (
        state["hist2d"]["Ej_Vs_T_Diff_Corrected"],
        edges["corrected"],
        edges["energy"],
    )
    output_file["Ej_Vs_T_Diff"] = (
        state["hist2d"]["Ej_Vs_T_Diff"],
        edges["time"],
        edges["energy"],
    )
    output_file["T_Diff_Counts_Vs_Detector_Pair"] = (
        state["pair_counts"],
        edges["detector"],
        edges["detector"],
    )
    pair_average = np.divide(
        state["pair_sum_corrected"],
        state["pair_counts"],
        out=np.zeros_like(state["pair_sum_corrected"]),
        where=state["pair_counts"] > 0,
    )
    output_file["Average_T_Diff_Corrected_Vs_Detector_Pair"] = (
        pair_average,
        edges["detector"],
        edges["detector"],
    )


def write_offset_tables(
    output_file: Any,
    detector_offsets: np.ndarray,
    offset_pair_counts: np.ndarray,
    offset_pair_means: np.ndarray,
    min_pair_count: int,
) -> None:
    detector_edges = np.linspace(-0.5, 17.5, 19)
    output_file["Detector_Time_Offsets"] = detector_offsets, detector_edges

    used_pair_counts = np.zeros(18, dtype=np.int32)
    for detector in range(18):
        used_pair_counts[detector] = int(
            np.sum(offset_pair_counts[detector, :] >= min_pair_count)
            + np.sum(offset_pair_counts[:, detector] >= min_pair_count)
        )
    offset_tree = output_file.mktree(
        "DetectorOffsetCorrections",
        {
            "detector": np.int32,
            "offset": np.float64,
            "used_pair_directions": np.int32,
        },
    )
    offset_tree.extend(
        {
            "detector": np.arange(18, dtype=np.int32),
            "offset": detector_offsets.astype(np.float64),
            "used_pair_directions": used_pair_counts,
        }
    )

    pair_i, pair_j = np.nonzero(offset_pair_counts > 0)
    pair_tree = output_file.mktree(
        "DetectorOffsetPairEstimates",
        {
            "index_i": np.int32,
            "index_j": np.int32,
            "mean_model_corrected_tdiff": np.float64,
            "count": np.int64,
            "used_in_offset_fit": np.bool_,
        },
    )
    pair_tree.extend(
        {
            "index_i": pair_i.astype(np.int32),
            "index_j": pair_j.astype(np.int32),
            "mean_model_corrected_tdiff": offset_pair_means[pair_i, pair_j].astype(np.float64),
            "count": offset_pair_counts[pair_i, pair_j].astype(np.int64),
            "used_in_offset_fit": (offset_pair_counts[pair_i, pair_j] >= min_pair_count),
        }
    )


def create_tree(output_file: Any):
    return output_file.mktree(
        "TimeCorrection",
        {
            "file_index": np.int32,
            "entry": np.int64,
            "Ei": np.float64,
            "index_i": np.int32,
            "Ej": np.float64,
            "index_j": np.int32,
            "T_Diff": np.float64,
            "T_pred": np.float64,
            "T_Model_Corrected": np.float64,
            "T_Offset_Correction": np.float64,
            "T_Diff_Corrected": np.float64,
        },
    )


def fill_outputs(
    file_specs,
    model,
    chunk_size,
    total_entries,
    require_dynode,
    pair_builder_args,
    tree,
    state,
    detector_offsets,
    args,
):
    with tqdm(total=total_entries, unit="events", desc="Writing corrected output") as pbar:
        for completed, pair_df in iter_pair_chunks(
            file_specs,
            chunk_size,
            require_dynode,
            pair_builder_args,
        ):
            pbar.update(completed)
            if pair_df.empty:
                continue
            predictions = predict_tdiff(model, pair_df[FEATURE_COLUMNS])
            pair_df = pair_df.copy()
            pair_df["T_pred"] = predictions
            index_i = pair_df["index_i"].to_numpy(dtype=np.intp)
            index_j = pair_df["index_j"].to_numpy(dtype=np.intp)
            if hasattr(model, "pair_offset_correction"):
                offset_correction = model.pair_offset_correction(index_i, index_j, detector_offsets)
            else:
                offset_correction = detector_pair_offsets(index_i, index_j, detector_offsets)
            pair_df["T_Model_Corrected"] = pair_df["T_Diff"].to_numpy(dtype=float) - predictions
            pair_df["T_Offset_Correction"] = offset_correction
            pair_df["T_Diff_Corrected"] = pair_df["T_Model_Corrected"].to_numpy(dtype=float) - offset_correction
            if tree is not None:
                tree.extend(
                    {
                        "file_index": pair_df["file_index"].to_numpy(dtype=np.int32),
                        "entry": pair_df["entry"].to_numpy(dtype=np.int64),
                        "Ei": pair_df["Ei"].to_numpy(dtype=np.float64),
                        "index_i": pair_df["index_i"].to_numpy(dtype=np.int32),
                        "Ej": pair_df["Ej"].to_numpy(dtype=np.float64),
                        "index_j": pair_df["index_j"].to_numpy(dtype=np.int32),
                        "T_Diff": pair_df["T_Diff"].to_numpy(dtype=np.float64),
                        "T_pred": pair_df["T_pred"].to_numpy(dtype=np.float64),
                        "T_Model_Corrected": pair_df["T_Model_Corrected"].to_numpy(dtype=np.float64),
                        "T_Offset_Correction": pair_df["T_Offset_Correction"].to_numpy(dtype=np.float64),
                        "T_Diff_Corrected": pair_df["T_Diff_Corrected"].to_numpy(dtype=np.float64),
                    }
                )

            tdiff = pair_df["T_Diff"].to_numpy(dtype=float)
            corrected = pair_df["T_Diff_Corrected"].to_numpy(dtype=float)
            ei = pair_df["Ei"].to_numpy(dtype=float)
            ej = pair_df["Ej"].to_numpy(dtype=float)

            add_hist2d(state, "Ej_Vs_T_Diff_Corrected", corrected, ej, "corrected", "energy")
            add_hist2d(state, "Ej_Vs_T_Diff", tdiff, ej, "time", "energy")
            np.add.at(state["pair_counts"], (index_i, index_j), 1.0)
            np.add.at(state["pair_sum_corrected"], (index_i, index_j), corrected)

            prompt_mask = (
                (args.prompt_low <= ei)
                & (ei <= args.prompt_high)
                & (args.prompt_low <= ej)
                & (ej <= args.prompt_high)
            )
            add_hist1d(state, "Prompt_Response_Corrected", corrected[prompt_mask], "corrected")
            add_hist1d(state, "Prompt_Response", tdiff[prompt_mask], "time")

            reference_mask = index_i == args.reference_detector
            for det in np.unique(index_j[reference_mask]):
                if det == args.reference_detector:
                    continue
                det_mask = reference_mask & (index_j == det)
                before_name = f"Labr{args.reference_detector}_minus_Labr{int(det)}_Before"
                after_name = f"Labr{args.reference_detector}_minus_Labr{int(det)}_After"
                add_hist1d(state, before_name, tdiff[det_mask], "time")
                add_hist1d(state, after_name, corrected[det_mask], "corrected")


def write_file_map(file_specs: list[tuple[Path, str, int]], output_path: Path) -> Path:
    map_path = output_path.with_suffix(".files.txt")
    with open(map_path, "w") as handle:
        for file_index, (input_path, _, _) in enumerate(file_specs):
            handle.write(f"{file_index} {input_path}\n")
    return map_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply calibrated-energy timing corrections to raw ROOT runs."
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
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Timing model joblib file from test_6D.py.",
    )
    parser.add_argument(
        "--coefficients",
        type=Path,
        default=DEFAULT_COEFFICIENTS,
        help="LaBr energy calibration coefficients CSV.",
    )
    parser.add_argument("--stop", type=int, help="Maximum entries per file to read.")
    parser.add_argument("--chunk-size", default="100 MB", help="uproot chunk size.")
    parser.add_argument("--output", type=Path, help="Output ROOT file. Default includes run label.")
    parser.add_argument("--min-ecal", type=float, default=100.0)
    parser.add_argument("--min-time", type=float, default=100.0)
    parser.add_argument("--min-dynode-time", type=float, default=100.0)
    parser.add_argument("--no-dynode-gate", action="store_true")
    parser.add_argument("--time-bin-width", type=float, default=0.01)
    parser.add_argument("--energy-bin-width", type=float, default=10.0)
    parser.add_argument("--prompt-low", type=float, default=494.55)
    parser.add_argument("--prompt-high", type=float, default=527.45)
    parser.add_argument("--reference-detector", type=int, default=8)
    parser.add_argument(
        "--no-offset-correction",
        action="store_true",
        help="Disable run-by-run detector offset solving.",
    )
    parser.add_argument(
        "--offset-range",
        nargs=2,
        type=float,
        default=(-20.0, 20.0),
        metavar=("LOW", "HIGH"),
        help="Model-corrected residual range used to estimate detector offsets.",
    )
    parser.add_argument(
        "--min-offset-pair-count",
        type=int,
        default=100,
        help="Minimum pair entries needed before a pair contributes to offset solving.",
    )
    parser.add_argument(
        "--time-range",
        nargs=2,
        type=float,
        metavar=("LOW", "HIGH"),
        help="Manual T_Diff histogram range.",
    )
    parser.add_argument(
        "--corrected-time-range",
        nargs=2,
        type=float,
        metavar=("LOW", "HIGH"),
        help="Manual T_Diff_Corrected histogram range.",
    )
    parser.add_argument(
        "--energy-range",
        nargs=2,
        type=float,
        metavar=("LOW", "HIGH"),
        help="Manual Ej histogram range.",
    )
    parser.add_argument(
        "--max-time-bins",
        type=int,
        default=20000,
        help="Maximum time-axis bins when ranges are chosen automatically.",
    )
    parser.add_argument(
        "--max-energy-bins",
        type=int,
        default=5000,
        help="Maximum energy-axis bins when ranges are chosen automatically.",
    )
    parser.add_argument(
        "--max-2d-bins",
        type=int,
        default=20000000,
        help="Maximum bins in each 2D diagnostic histogram.",
    )
    parser.add_argument(
        "--no-tree",
        action="store_true",
        help="Write only histograms. This is much faster for large runs.",
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=1,
        help="ROOT output ZLIB compression level. Use 0 for fastest/largest output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.time_bin_width <= 0:
        raise ValueError("--time-bin-width must be greater than 0")
    if args.energy_bin_width <= 0:
        raise ValueError("--energy-bin-width must be greater than 0")
    if not 0 <= args.reference_detector < 18:
        raise ValueError("--reference-detector must be between 0 and 17")
    if not 0 <= args.compression_level <= 9:
        raise ValueError("--compression-level must be between 0 and 9")
    if args.min_offset_pair_count < 1:
        raise ValueError("--min-offset-pair-count must be at least 1")
    if args.max_time_bins <= 0:
        raise ValueError("--max-time-bins must be greater than 0")
    if args.max_energy_bins <= 0:
        raise ValueError("--max-energy-bins must be greater than 0")
    if args.max_2d_bins <= 0:
        raise ValueError("--max-2d-bins must be greater than 0")

    input_patterns, percentage = split_inputs_and_percentage(args.inputs)
    input_paths = expand_input_patterns(input_patterns)
    fraction = percentage_to_fraction(percentage)
    output_path = args.output or default_output_path(input_paths)
    require_dynode = not args.no_dynode_gate

    intercepts, slopes, has_coeff = load_calibration_coefficients(args.coefficients)
    model = load_timing_model(args.model)
    allowed_i, allowed_j = allowed_model_detectors(model)
    if allowed_i is not None and allowed_j is not None:
        print(
            "Restricting pairs to detector indices known by the model: "
            f"i={sorted(allowed_i)}, j={sorted(allowed_j)}"
        )

    file_specs = collect_file_specs(input_paths, args.tree, fraction, args.stop, require_dynode)
    total_entries = sum(entry_stop for _, _, entry_stop in file_specs)
    pair_builder_args = (
        intercepts,
        slopes,
        has_coeff,
        args.min_ecal,
        args.min_time,
        require_dynode,
        args.min_dynode_time,
        allowed_i,
        allowed_j,
    )

    detector_offsets, offset_pair_counts, offset_pair_means = estimate_detector_offsets(
        file_specs,
        model,
        args.chunk_size,
        total_entries,
        require_dynode,
        pair_builder_args,
        args,
    )
    tdiff_range, corrected_range, ej_range = scan_ranges(
        file_specs,
        model,
        args.chunk_size,
        total_entries,
        require_dynode,
        pair_builder_args,
        detector_offsets,
    )
    tdiff_range, corrected_range, ej_range = apply_histogram_range_limits(
        tdiff_range,
        corrected_range,
        ej_range,
        args,
    )
    compression = None if args.compression_level == 0 else uproot.ZLIB(args.compression_level)
    histogram_state = create_histogram_state(tdiff_range, corrected_range, ej_range, args)
    output_file = uproot.recreate(output_path, compression=compression)
    try:
        tree = None if args.no_tree else create_tree(output_file)
        if args.no_tree:
            print("Skipping TimeCorrection tree; writing histogram diagnostics only.", flush=True)
        else:
            print("Writing TimeCorrection tree and histogram diagnostics.", flush=True)
        fill_outputs(
            file_specs,
            model,
            args.chunk_size,
            total_entries,
            require_dynode,
            pair_builder_args,
            tree,
            histogram_state,
            detector_offsets,
            args,
        )
        print("Writing histogram objects...", flush=True)
        write_histograms(output_file, histogram_state)
        print("Writing detector offset summary...", flush=True)
        write_offset_tables(
            output_file,
            detector_offsets,
            offset_pair_counts,
            offset_pair_means,
            args.min_offset_pair_count,
        )
        print("Closing output ROOT file...", flush=True)
    finally:
        output_file.close()

    file_map = write_file_map(file_specs, output_path)

    print(f"Wrote {output_path}")
    print(f"Wrote {file_map}")


if __name__ == "__main__":
    # Make classes available under __main__ for joblib models saved by test_6D.py.
    sys.modules["__main__"].Log1pTransformer = Log1pTransformer
    sys.modules["__main__"].Expm1Transformer = Expm1Transformer
    main()
