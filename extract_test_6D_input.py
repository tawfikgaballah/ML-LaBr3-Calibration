"""Build the pandas inputs used by test_6D.py from a ROOT file.

This script mirrors the active event/pair selection in ``Timing_Branches 1.C``
and writes the dataframe columns consumed by ``test_6D.py``:

    Ei, index_i, Ej, index_j, tdiff, tdiff_aligned

It reads one or more ROOT file paths or glob patterns from the command line,
then optionally reads only a requested percentage of each input tree.
"""

from __future__ import annotations

import argparse
import csv
import glob
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm


TEST_6D_COLUMNS = ["Ei", "index_i", "Ej", "index_j", "tdiff", "tdiff_aligned"]
DEFAULT_COEFFICIENTS = Path("labr_energy_calibration_coefficients.csv")
RAW_BRANCHES = {
    "labr3_cfdfailbit": "rootout/labr/labr.cfdfailbit[18]",
    "labr3_energy": "rootout/labr/labr.energy[18]",
    "labr3_time": "rootout/labr/labr.time[18]",
    "pspmt_dycfdfailbit": "rootout/NpspmtCeBr/NpspmtCeBr.dycfdfailbit",
    "pspmt_dytime": "rootout/NpspmtCeBr/NpspmtCeBr.dytime",
}


def import_uproot():
    try:
        import uproot
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "This script needs uproot to read ROOT files. Run it in the same "
            "Python environment you use for ROOT/uproot analysis, or install "
            "uproot there."
        ) from exc
    return uproot


def first_existing_tree(root_file: Any, preferred: str | None) -> str:
    """Return the requested tree, or a sensible tree name from the file."""
    if preferred:
        if preferred in root_file:
            return preferred
        if f"{preferred};1" in root_file:
            return f"{preferred};1"
        raise KeyError(f"Tree {preferred!r} was not found in {root_file.file_path}")

    for candidate in ("TOutput", "T", "Tree", "tree"):
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
        raise KeyError(f"No TTree objects found in {root_file.file_path}")
    return tree_names[0]


def scalar(value: Any) -> float:
    """Convert scalar or one-element ROOT branch values to a Python number."""
    array = np.asarray(value)
    if array.ndim == 0:
        return array.item()
    if array.size == 0:
        return np.nan
    return array.reshape(-1)[0].item()


def percentage_to_fraction(value: float | None) -> float | None:
    """Convert a user percentage to a fraction of entries to read."""
    if value is None:
        return None
    if value <= 0:
        raise ValueError("percentage must be greater than 0")
    if value <= 100:
        return value / 100.0
    raise ValueError("percentage must be at most 100")


def looks_like_number(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def split_input_patterns_and_percentage(
    values: list[str],
) -> tuple[list[str], float | None]:
    """Treat the final positional number as a percentage."""
    if not values:
        raise ValueError("at least one ROOT file or glob pattern is required")
    if looks_like_number(values[-1]):
        if len(values) == 1:
            raise ValueError("a ROOT file or glob pattern is required before percentage")
        return values[:-1], float(values[-1])
    return values, None


def expand_input_patterns(patterns: list[str]) -> list[Path]:
    """Expand shell-style globs while preserving literal file paths."""
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


def default_output_prefix(input_paths: list[Path]) -> Path:
    return Path(f"test_6D_input_{run_label_from_paths(input_paths)}")


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

    missing = [str(det) for det in range(18) if not has_coeff[det]]
    if missing:
        print(
            "Warning: skipping detector(s) without coefficients: "
            + ", ".join(missing)
        )
    return intercepts, slopes, has_coeff


def load_existing_toutput(
    input_path: Path,
    tree_name: str,
    stop: int | None,
) -> pd.DataFrame:
    """Read a prebuilt TOutput tree into the exact test_6D.py column order."""
    uproot = import_uproot()
    with uproot.open(input_path) as root_file:
        tree = root_file[tree_name]
        missing = [name for name in TEST_6D_COLUMNS if name not in tree.keys()]
        if missing:
            raise KeyError(
                f"Tree {tree_name!r} is missing test_6D.py branches: {missing}"
            )
        return tree.arrays(TEST_6D_COLUMNS, library="pd", entry_stop=stop)[
            TEST_6D_COLUMNS
        ]


def build_from_raw_branches(
    input_path: Path,
    tree_name: str,
    stop: int | None,
    chunk_size: str,
    total_entries: int | None,
    intercepts: np.ndarray,
    slopes: np.ndarray,
    has_coeff: np.ndarray,
) -> pd.DataFrame:
    """Recreate the active Timing_Branches.C pair loop as a dataframe."""
    uproot = import_uproot()
    rows: list[pd.DataFrame] = []

    with uproot.open(input_path) as root_file:
        tree = root_file[tree_name]
        missing = [name for name in RAW_BRANCHES.values() if name not in tree.keys()]
        if missing:
            raise KeyError(f"Tree {tree_name!r} is missing raw branches: {missing}")

        iterator = tree.iterate(
            list(RAW_BRANCHES.values()),
            library="np",
            entry_stop=stop,
            step_size=chunk_size,
        )

        with tqdm(total=total_entries, unit="events", desc="Extracting") as pbar:
            for arrays in iterator:
                rows.append(
                    build_chunk_from_raw_arrays(arrays, intercepts, slopes, has_coeff)
                )
                pbar.update(len(arrays[RAW_BRANCHES["labr3_energy"]]))

    rows = [frame for frame in rows if not frame.empty]
    if not rows:
        return pd.DataFrame(columns=TEST_6D_COLUMNS)
    return pd.concat(rows, ignore_index=True)


def build_chunk_from_raw_arrays(
    dict_arrays: dict[str, np.ndarray],
    intercepts: np.ndarray,
    slopes: np.ndarray,
    has_coeff: np.ndarray,
) -> pd.DataFrame:
    """Build one dataframe chunk from raw branch arrays."""
    chunk_rows: list[tuple[float, int, float, int, float, float]] = []
    n_events = len(dict_arrays[RAW_BRANCHES["labr3_energy"]])

    for event_index in range(n_events):
        cfdfail = np.asarray(
            dict_arrays[RAW_BRANCHES["labr3_cfdfailbit"]][event_index]
        )
        raw_energy = np.asarray(
            dict_arrays[RAW_BRANCHES["labr3_energy"]][event_index], dtype=float
        )
        time = np.asarray(
            dict_arrays[RAW_BRANCHES["labr3_time"]][event_index], dtype=float
        )
        dyfail = scalar(
            dict_arrays[RAW_BRANCHES["pspmt_dycfdfailbit"]][event_index]
        )
        dytime = scalar(dict_arrays[RAW_BRANCHES["pspmt_dytime"]][event_index])

        if  dytime <=100:
            continue

        detector_count = min(len(raw_energy), len(time), len(cfdfail), len(has_coeff))
        ecal = intercepts[:detector_count] + slopes[:detector_count] * raw_energy[:detector_count]
        for index_i in range(detector_count):
            if (
                # cfdfail[index_i] == 1
                not has_coeff[index_i]
                or raw_energy[index_i] <= 0
                or ecal[index_i] <= 10
                or ecal[index_i] > 2000
                or time[index_i] <= 100
            ):
                continue

            for index_j in range(detector_count):
                if (
                    # cfdfail[index_j] == 1
                    not has_coeff[index_j]
                    or raw_energy[index_j] <= 0
                    or ecal[index_j] <= 10
                    or ecal[index_j] > 2000
                    or time[index_j] <= 100
                ):
                    continue

                tdiff = time[index_i] - time[index_j]
                if -1000 < tdiff < 1000:
                    tdiff_aligned = tdiff if index_i == 0 else 0.0
                    chunk_rows.append(
                            (
                                ecal[index_i],
                                index_i,
                                ecal[index_j],
                                index_j,
                                tdiff,
                                tdiff_aligned,
                            )
                )

    return pd.DataFrame(chunk_rows, columns=TEST_6D_COLUMNS)


def make_test_6d_frames(
    df: pd.DataFrame,
    tdiff_window: tuple[float, float],
    zscore_limit: float,
) -> dict[str, pd.DataFrame | pd.Series]:
    """Create the same top-level dataframes/series that test_6D.py prepares."""
    low, high = tdiff_window
    df_filtered = df[df["tdiff"].between(low, high)].copy()

    features = df.drop("tdiff", axis=1)
    target = df["tdiff"]

    target_std = target.std()
    if target_std and not np.isnan(target_std):
        z_scores = (target - target.mean()) / target_std
        outliers = z_scores.abs() > zscore_limit
    else:
        outliers = pd.Series(False, index=df.index)

    return {
        "df": df,
        "df_filtered": df_filtered,
        "features": features,
        "target": target,
        "outliers_df": df[outliers].copy(),
        "filtered_data": df[~outliers].copy(),
        "X": df_filtered.drop("tdiff_aligned", axis=1),
        "y": df_filtered["tdiff_aligned"],
    }


def write_outputs(frames: dict[str, Any], output_prefix: Path) -> None:
    """Write pickle outputs plus CSV copies for easy inspection."""
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    pickle_path = output_prefix.with_suffix(".pkl")
    frames["df"].to_csv(output_prefix.with_name(f"{output_prefix.name}_df.csv"), index=False)
    frames["df_filtered"].to_csv(
        output_prefix.with_name(f"{output_prefix.name}_df_filtered.csv"),
        index=False,
    )
    pd.to_pickle(frames, pickle_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract ROOT input into the pandas dataframes expected by test_6D.py."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help=(
            "One or more raw input ROOT files or glob patterns. If the final "
            "argument is numeric, it is treated as percentage. Use 1 for 1%%, "
            "10 for 10%%, or 0.01 for 0.01%%."
        ),
    )
    parser.add_argument(
        "--tree",
        help="Input tree name. Defaults to TOutput, T, Tree, tree, then first TTree.",
    )
    parser.add_argument(
        "--from-toutput",
        action="store_true",
        help="Read an existing TOutput-style tree instead of rebuilding from raw branches.",
    )
    parser.add_argument(
        "--coefficients",
        type=Path,
        default=DEFAULT_COEFFICIENTS,
        help=(
            "LaBr energy calibration coefficients CSV. Raw labr.energy is "
            "converted to Ei/Ej with Ecal = intercept + slope * energy. "
            f"Default: {DEFAULT_COEFFICIENTS}."
        ),
    )
    parser.add_argument(
        "--stop",
        type=int,
        help="Maximum number of input tree entries/events to read.",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        help=(
            "Fraction of tree entries/events to read, matching test_6D.py's "
            "0.01 style. Overrides the positional percentage."
        ),
    )
    parser.add_argument(
        "--chunk-size",
        default="100 MB",
        help="uproot chunk size for raw extraction. Default: 100 MB.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        help=(
            "Output prefix for .pkl and .csv files. Default includes the "
            "input run label, e.g. test_6D_input_run-0989."
        ),
    )
    parser.add_argument("--tdiff-min", type=float, default=-100.0)
    parser.add_argument("--tdiff-max", type=float, default=100.0)
    parser.add_argument("--zscore-limit", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    uproot = import_uproot()
    input_patterns, percentage = split_input_patterns_and_percentage(args.inputs)
    input_paths = expand_input_patterns(input_patterns)
    output_prefix = args.output_prefix or default_output_prefix(input_paths)
    dataframes: list[pd.DataFrame] = []
    source_modes: set[str] = set()
    if args.from_toutput:
        intercepts = slopes = has_coeff = None
    else:
        intercepts, slopes, has_coeff = load_calibration_coefficients(args.coefficients)
        print(f"Calculating Ei/Ej from raw labr.energy using {args.coefficients}")

    for input_path in input_paths:
        with uproot.open(input_path) as root_file:
            tree_name = first_existing_tree(root_file, args.tree)
            total_entries = root_file[tree_name].num_entries
            stop = args.stop
            fraction = args.fraction
            if fraction is None:
                fraction = percentage_to_fraction(percentage)
            if fraction is not None:
                if not 0 < fraction <= 1:
                    raise ValueError("--fraction must be greater than 0 and at most 1")
                fraction_stop = max(1, int(total_entries * fraction))
                stop = min(stop, fraction_stop) if stop is not None else fraction_stop
            entries_to_read = (
                min(stop, total_entries) if stop is not None else total_entries
            )

            branch_names = set(root_file[tree_name].keys())
            has_toutput_shape = all(name in branch_names for name in TEST_6D_COLUMNS)

        if args.from_toutput:
            if not has_toutput_shape:
                raise KeyError(
                    f"Tree {tree_name!r} in {input_path} does not contain the "
                    f"TOutput branches: {TEST_6D_COLUMNS}"
                )
            file_df = load_existing_toutput(input_path, tree_name, stop)
            source_mode = "existing TOutput-style tree"
        else:
            print(f"Reading {input_path}:{tree_name}")
            file_df = build_from_raw_branches(
                input_path,
                tree_name,
                stop,
                args.chunk_size,
                entries_to_read,
                intercepts,
                slopes,
                has_coeff,
            )
            source_mode = "raw detector branches"

        dataframes.append(file_df)
        source_modes.add(source_mode)

    df = (
        pd.concat(dataframes, ignore_index=True)
        if dataframes
        else pd.DataFrame(columns=TEST_6D_COLUMNS)
    )
    source_mode = ", ".join(sorted(source_modes))

    frames = make_test_6d_frames(
        df,
        tdiff_window=(args.tdiff_min, args.tdiff_max),
        zscore_limit=args.zscore_limit,
    )
    write_outputs(frames, output_prefix)

    print(f"Read {source_mode} from {len(input_paths)} file(s)")
    print(f"Rows in df: {len(frames['df'])}")
    print(f"Rows in df_filtered: {len(frames['df_filtered'])}")
    print(f"Wrote {output_prefix.with_suffix('.pkl')}")
    print(f"Wrote {output_prefix.name}_df.csv")
    print(f"Wrote {output_prefix.name}_df_filtered.csv")


if __name__ == "__main__":
    main()
