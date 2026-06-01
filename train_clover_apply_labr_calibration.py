# Author: Tawfik Gaballah
# GitHub: tawfikgaballah
# Project: ML-LaBr3-Calibration

"""Train clover.energy -> clover.ecal and apply it to labr.energy.

The learned model is a one-dimensional calibration function:

    raw energy -> calibrated energy

It is trained from calibrated clover data and then used to predict calibrated
LaBr energies from raw LaBr energies.
"""

from __future__ import print_function

import argparse
import glob
from pathlib import Path

import joblib
import numpy as np
import uproot
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from tqdm import tqdm


CLOVER_ENERGY_BRANCH = "rootout/clover/clover.energy[64]"
CLOVER_ECAL_BRANCH = "rootout/clover/clover.ecal[64]"
LABR_ENERGY_BRANCH = "rootout/labr/labr.energy[18]"


def looks_like_number(value):
    try:
        float(value)
    except ValueError:
        return False
    return True


def percentage_to_fraction(value):
    if value is None:
        return None
    if value <= 0:
        raise ValueError("percentage must be greater than 0")
    if value <= 100:
        return value / 100.0
    raise ValueError("percentage must be at most 100")


def split_inputs_and_percentage(values):
    if not values:
        raise ValueError("at least one ROOT file or glob pattern is required")
    if looks_like_number(values[-1]):
        if len(values) == 1:
            raise ValueError("a ROOT file or glob pattern is required before percentage")
        return values[:-1], float(values[-1])
    return values, None


def expand_input_patterns(patterns):
    paths = []
    for pattern in patterns:
        if any(char in pattern for char in "*?["):
            matches = sorted(glob.glob(pattern))
            if not matches:
                raise FileNotFoundError("No files matched pattern: {}".format(pattern))
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(pattern))

    unique_paths = []
    seen = set()
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique_paths.append(path)
    return unique_paths


def first_existing_tree(root_file, preferred):
    if preferred:
        if preferred in root_file:
            return preferred
        if "{};1".format(preferred) in root_file:
            return "{};1".format(preferred)
        raise KeyError("Tree {!r} was not found".format(preferred))

    for candidate in ("data", "T", "Tree", "tree"):
        if candidate in root_file:
            return candidate
        if "{};1".format(candidate) in root_file:
            return "{};1".format(candidate)

    tree_names = [
        key
        for key in root_file.keys()
        if getattr(root_file[key], "classname", "").startswith("TTree")
    ]
    if not tree_names:
        raise KeyError("No TTree objects found")
    return tree_names[0]


def entries_to_read(tree, fraction, stop):
    total_entries = tree.num_entries
    selected = total_entries
    if fraction is not None:
        selected = max(1, int(total_entries * fraction))
    if stop is not None:
        selected = min(selected, stop)
    return selected


def collect_file_specs(input_paths, tree_name_arg, fraction, stop):
    file_specs = []
    required = [CLOVER_ENERGY_BRANCH, CLOVER_ECAL_BRANCH, LABR_ENERGY_BRANCH]
    for input_path in input_paths:
        with uproot.open(input_path) as root_file:
            tree_name = first_existing_tree(root_file, tree_name_arg)
            tree = root_file[tree_name]
            missing = [name for name in required if name not in tree.keys()]
            if missing:
                raise KeyError("{}:{} is missing {}".format(input_path, tree_name, missing))
            file_specs.append((input_path, tree_name, entries_to_read(tree, fraction, stop)))
    return file_specs


def collect_training_pairs(file_specs, chunk_size, max_train_points):
    raw_chunks = []
    cal_chunks = []
    collected = 0
    total_entries = sum(entry_stop for _, _, entry_stop in file_specs)

    with tqdm(total=total_entries, unit="events", desc="Collecting clover training") as pbar:
        for input_path, tree_name, entry_stop in file_specs:
            with uproot.open(input_path) as root_file:
                tree = root_file[tree_name]
                for arrays in tree.iterate(
                    [CLOVER_ENERGY_BRANCH, CLOVER_ECAL_BRANCH],
                    library="np",
                    entry_stop=entry_stop,
                    step_size=chunk_size,
                ):
                    raw = arrays[CLOVER_ENERGY_BRANCH].reshape(-1)
                    cal = arrays[CLOVER_ECAL_BRANCH].reshape(-1)
                    pbar.update(len(arrays[CLOVER_ENERGY_BRANCH]))

                    valid = np.isfinite(raw) & np.isfinite(cal) & (raw > 0) & (cal > 0)
                    raw = raw[valid]
                    cal = cal[valid]
                    if raw.size == 0:
                        continue

                    room = max_train_points - collected
                    if room <= 0:
                        break
                    if raw.size > room:
                        raw = raw[:room]
                        cal = cal[:room]

                    raw_chunks.append(raw)
                    cal_chunks.append(cal)
                    collected += raw.size
                if collected >= max_train_points:
                    break

    if not raw_chunks:
        raise ValueError("No valid clover.energy/clover.ecal training pairs found")

    raw_energy = np.concatenate(raw_chunks).reshape(-1, 1)
    calibrated_energy = np.concatenate(cal_chunks)
    return raw_energy, calibrated_energy


def train_model(raw_energy, calibrated_energy, degree, alpha):
    model = Pipeline(
        [
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )
    model.fit(raw_energy, calibrated_energy)
    return model


def write_labr_predictions(file_specs, model, output_path, chunk_size):
    branch_types = {
        "file_index": np.int32,
        "entry": np.int64,
        "detector": np.int32,
        "labr_energy": np.float64,
        "labr_ecal_pred": np.float64,
    }
    total_entries = sum(entry_stop for _, _, entry_stop in file_specs)

    with uproot.recreate(output_path) as output_file:
        tree_out = output_file.mktree("LaBrCalibrationML", branch_types)

        with tqdm(total=total_entries, unit="events", desc="Predicting LaBr ecal") as pbar:
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
                        labr_energy = arrays[LABR_ENERGY_BRANCH][:, :18]
                        entries_in_chunk = len(labr_energy)
                        pbar.update(entries_in_chunk)

                        valid = np.isfinite(labr_energy) & (labr_energy > 0)
                        if not np.any(valid):
                            entry_offset += entries_in_chunk
                            continue

                        event_indices, detector_indices = np.where(valid)
                        raw_values = labr_energy[valid].astype(np.float64)
                        predicted = model.predict(raw_values.reshape(-1, 1))

                        tree_out.extend(
                            {
                                "file_index": np.full(raw_values.size, file_index, dtype=np.int32),
                                "entry": (event_indices + entry_offset).astype(np.int64),
                                "detector": detector_indices.astype(np.int32),
                                "labr_energy": raw_values.astype(np.float64),
                                "labr_ecal_pred": predicted.astype(np.float64),
                            }
                        )
                        entry_offset += entries_in_chunk


def write_file_map(file_specs, output_path):
    map_path = Path(output_path).with_suffix(".files.txt")
    with open(map_path, "w") as handle:
        for file_index, (input_path, _, _) in enumerate(file_specs):
            handle.write("{} {}\n".format(file_index, input_path))
    return map_path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train clover.energy -> clover.ecal, then apply the calibration "
            "function to labr.energy."
        )
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
    parser.add_argument("--chunk-size", default="100 MB", help="uproot chunk size.")
    parser.add_argument("--degree", type=int, default=2, help="Polynomial degree.")
    parser.add_argument("--alpha", type=float, default=1.0, help="Ridge alpha.")
    parser.add_argument(
        "--max-train-points",
        type=int,
        default=2000000,
        help="Maximum clover points to keep in memory for training.",
    )
    parser.add_argument(
        "--model-output",
        default="clover_energy_to_ecal_model.joblib",
        help="Output joblib model path.",
    )
    parser.add_argument(
        "--predictions-output",
        default="labr_ecal_from_clover_model.root",
        help="Output ROOT file with predicted LaBr calibrated energies.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_patterns, percentage = split_inputs_and_percentage(args.inputs)
    input_paths = expand_input_patterns(input_patterns)
    fraction = percentage_to_fraction(percentage)
    file_specs = collect_file_specs(input_paths, args.tree, fraction, args.stop)

    raw_energy, calibrated_energy = collect_training_pairs(
        file_specs,
        args.chunk_size,
        args.max_train_points,
    )
    print("Training pairs: {}".format(len(calibrated_energy)))

    model = train_model(raw_energy, calibrated_energy, args.degree, args.alpha)
    joblib.dump(model, args.model_output)
    print("Wrote {}".format(args.model_output))

    write_labr_predictions(file_specs, model, args.predictions_output, args.chunk_size)
    file_map = write_file_map(file_specs, args.predictions_output)
    print("Wrote {}".format(args.predictions_output))
    print("Wrote {}".format(file_map))


if __name__ == "__main__":
    main()
