# Author: Tawfik Gaballah
# GitHub: tawfikgaballah
# Project: ML-LaBr3-Calibration

"""Interactively calibrate LaBr raw-energy histograms.

This script reads the ``labr_energy_det0`` ... ``labr_energy_det17`` histograms
created by ``draw_labr_histograms.py``. For each detector, it asks whether to
calibrate, lets you click peak locations, fits each peak with a Gaussian plus a
linear background, asks for the known gamma energy, and then builds a linear
calibration for that detector.
"""

from __future__ import annotations

import argparse
import csv
import re
import time
from pathlib import Path
from typing import Any

import numpy as np


CLICK_HELPER_CODE = r"""
namespace LaBrClick {
  bool clicked = false;
  double x = 0.0;

  void Reset() {
    clicked = false;
    x = 0.0;
  }

  bool Clicked() {
    return clicked;
  }

  double X() {
    return x;
  }

  void Handler() {
    if (gPad == nullptr) return;
    int event = gPad->GetEvent();
    if (event != kButton1Down) return;
    int px = gPad->GetEventX();
    double xpad = gPad->AbsPixeltoX(px);
    x = gPad->PadtoX(xpad);
    clicked = true;
  }
}
"""


def import_root():
    try:
        import ROOT
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "This script needs PyROOT. On the analysis server, load the ROOT "
            "environment/module before running it."
        ) from exc
    return ROOT


def yes_no(prompt: str, default: bool = True) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        answer = input(f"{prompt} {suffix}: ").strip().lower()
        if not answer:
            return default
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("Please answer y or n.")


def ask_int(prompt: str, minimum: int = 0) -> int:
    while True:
        try:
            value = int(input(f"{prompt}: ").strip())
        except ValueError:
            print("Please enter an integer.")
            continue
        if value < minimum:
            print(f"Please enter a value >= {minimum}.")
            continue
        return value


def ask_float(prompt: str) -> float:
    while True:
        try:
            return float(input(f"{prompt}: ").strip())
        except ValueError:
            print("Please enter a number.")


def parse_gamma_energies(text: str) -> list[float]:
    values = [part.strip() for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("at least one gamma energy is required")
    return [float(value) for value in values]


def get_gamma_energies(args: argparse.Namespace) -> list[float]:
    if args.gamma_energies:
        return parse_gamma_energies(args.gamma_energies)

    n_peaks = ask_int("How many gamma peaks do you want to calibrate on", 1)
    gamma_energies: list[float] = []
    for peak_index in range(n_peaks):
        gamma_energies.append(ask_float(f"Gamma energy for peak {peak_index + 1}"))
    return gamma_energies


def parse_detectors(detector_text: str) -> list[int]:
    detectors: list[int] = []
    for part in detector_text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            detectors.extend(range(int(start_text), int(end_text) + 1))
        else:
            detectors.append(int(part))
    return sorted(set(detectors))


def run_label_from_path(path: Path) -> str:
    match = re.search(r"(run[-_]?\d+)", path.name, flags=re.IGNORECASE)
    if match:
        return match.group(1).replace("_", "-")
    return path.stem


def apply_default_output_paths(args: argparse.Namespace) -> None:
    run_label = run_label_from_path(args.input)
    if args.output is None:
        args.output = Path(f"labr_energy_calibrated_{run_label}.root")
    if args.calibration_points is None:
        args.calibration_points = f"labr_energy_calibration_points_{run_label}.csv"
    if args.coefficients is None:
        args.coefficients = f"labr_energy_calibration_coefficients_{run_label}.csv"


def load_histograms(root: Any, input_path: Path, detectors: list[int]) -> dict[int, Any]:
    root_file = root.TFile.Open(str(input_path), "READ")
    if not root_file or root_file.IsZombie():
        raise OSError(f"Could not open {input_path}")

    histograms: dict[int, Any] = {}
    for det in detectors:
        hist = root_file.Get(f"labr_energy_det{det}")
        if not hist:
            print(f"Detector {det}: missing histogram labr_energy_det{det}; skipping")
            continue
        if hist.InheritsFrom("TH2"):
            print(f"Detector {det}: labr_energy_det{det} is not a 1D histogram; skipping")
            continue
        if hist.GetEntries() <= 0 and hist.Integral() <= 0:
            print(f"Detector {det}: labr_energy_det{det} is empty; skipping")
            continue
        hist.SetDirectory(0)
        histograms[det] = hist

    root_file.Close()
    return histograms


def install_click_handler(root: Any) -> None:
    root.gInterpreter.Declare(CLICK_HELPER_CODE)


def wait_for_click(
    root: Any,
    canvas: Any,
    hist: Any,
    prompt: str = "Click near the peak center in the ROOT canvas...",
) -> float:
    canvas.cd()
    hist.Draw()
    canvas.Update()
    input(
        "Zoom or pan the ROOT canvas as needed, then press Enter here to "
        "enable peak clicking..."
    )
    canvas.cd()
    canvas.Update()
    canvas.DeleteExec("labr_peak_click")
    canvas.AddExec("labr_peak_click", "LaBrClick::Handler();")
    root.LaBrClick.Reset()
    print(prompt)
    while not root.LaBrClick.Clicked():
        root.gSystem.ProcessEvents()
        time.sleep(0.05)
    canvas.DeleteExec("labr_peak_click")
    clicked_x = float(root.LaBrClick.X())
    print(f"Clicked x = {clicked_x:.6g}")
    return clicked_x


def fit_peak(root: Any, hist: Any, clicked_x: float, fit_half_width: float):
    x_axis = hist.GetXaxis()
    fit_low = max(x_axis.GetXmin(), clicked_x - fit_half_width)
    fit_high = min(x_axis.GetXmax(), clicked_x + fit_half_width)
    peak_bin = hist.FindBin(clicked_x)
    peak_height = max(hist.GetBinContent(peak_bin), 1.0)
    bin_width = x_axis.GetBinWidth(peak_bin)
    sigma_guess = max(fit_half_width / 5.0, bin_width)

    fit = root.TF1(
        f"fit_{hist.GetName()}_{int(time.time() * 1000)}",
        "pol1(0)+gaus(2)",
        fit_low,
        fit_high,
    )
    fit.SetParameters(0.0, 0.0, peak_height, clicked_x, sigma_guess)
    fit.SetParNames("bg0", "bg1", "amp", "mean", "sigma")
    fit.SetParLimits(2, 0.0, peak_height * 100.0)
    fit.SetParLimits(3, fit_low, fit_high)
    fit.SetParLimits(4, max(bin_width * 0.25, 1e-6), fit_half_width)
    result = hist.Fit(fit, "RQS")
    centroid = float(fit.GetParameter(3))
    centroid_error = float(fit.GetParError(3))
    sigma = abs(float(fit.GetParameter(4)))
    return fit, result, fit_low, fit_high, centroid, centroid_error, sigma


def collect_detector_calibration(
    root: Any,
    canvas: Any,
    det: int,
    hist: Any,
    fit_half_width: float,
    gamma_energies: list[float],
) -> list[dict[str, float]]:
    points: list[dict[str, float]] = []

    for peak_index, gamma_energy in enumerate(gamma_energies):
        while True:
            print(
                f"\nDetector {det}, peak {peak_index + 1}/{len(gamma_energies)} "
                f"({gamma_energy:g})"
            )
            clicked_x = wait_for_click(root, canvas, hist)
            fit, result, fit_low, fit_high, centroid, centroid_error, sigma = fit_peak(
                root,
                hist,
                clicked_x,
                fit_half_width,
            )
            canvas.cd()
            hist.Draw()
            fit.Draw("same")
            canvas.Update()

            print(
                "Fit centroid = {:.6g} +/- {:.3g}, sigma = {:.6g}, status = {}".format(
                    centroid, centroid_error, sigma, int(result)
                )
            )
            answer = input("Accept fit? [Y/n/r=retry]: ").strip().lower()
            if answer == "r":
                continue
            if answer in ("n", "no"):
                print("Skipping this peak.")
                break

            points.append(
                {
                    "detector": det,
                    "peak_index": peak_index,
                    "centroid": centroid,
                    "centroid_error": centroid_error,
                    "sigma": sigma,
                    "gamma_energy": gamma_energy,
                    "fit_low": fit_low,
                    "fit_high": fit_high,
                }
            )
            break

    return points


def fit_linear_calibration(points: list[dict[str, float]]) -> dict[str, float]:
    centroids = np.array([point["centroid"] for point in points], dtype=float)
    gamma = np.array([point["gamma_energy"] for point in points], dtype=float)
    if len(points) == 1:
        slope = gamma[0] / centroids[0]
        intercept = 0.0
    else:
        slope, intercept = np.polyfit(centroids, gamma, 1)
    return {"slope": float(slope), "intercept": float(intercept), "n_peaks": len(points)}


def calibrated_hist_range(
    histograms: dict[int, Any],
    coefficients: dict[int, dict[str, float]],
    fallback: tuple[float, float] = (0.0, 4000.0),
) -> tuple[float, float]:
    values: list[float] = []
    for det, hist in histograms.items():
        if det not in coefficients:
            continue
        slope = coefficients[det]["slope"]
        intercept = coefficients[det]["intercept"]
        for bin_index in range(1, hist.GetNbinsX() + 1):
            if hist.GetBinContent(bin_index) <= 0:
                continue
            values.append(intercept + slope * hist.GetBinCenter(bin_index))

    if not values:
        return fallback
    low = min(values)
    high = max(values)
    if low == high:
        pad = max(abs(low) * 0.05, 1.0)
    else:
        pad = (high - low) * 0.05
    return low - pad, high + pad


def calibrate_histogram(
    root: Any,
    det: int,
    hist: Any,
    coeff: dict[str, float],
    bins: int,
    cal_range: tuple[float, float],
) -> Any:
    out = root.TH1D(
        f"labr_energy_cal_det{det}",
        f"Calibrated LaBr energy detector {det};Energy;Counts",
        bins,
        cal_range[0],
        cal_range[1],
    )
    slope = coeff["slope"]
    intercept = coeff["intercept"]
    for bin_index in range(1, hist.GetNbinsX() + 1):
        counts = hist.GetBinContent(bin_index)
        if counts <= 0:
            continue
        raw_energy = hist.GetBinCenter(bin_index)
        calibrated_energy = intercept + slope * raw_energy
        out.Fill(calibrated_energy, counts)
    return out


def make_detector_index_histogram(
    root: Any,
    name: str,
    title: str,
    calibrated_hists: dict[int, Any],
    bins: int,
    cal_range: tuple[float, float],
) -> Any:
    hist2d = root.TH2D(
        name,
        title,
        bins,
        cal_range[0],
        cal_range[1],
        18,
        0,
        18,
    )
    for det, hist in calibrated_hists.items():
        for bin_index in range(1, hist.GetNbinsX() + 1):
            counts = hist.GetBinContent(bin_index)
            if counts <= 0:
                continue
            hist2d.SetBinContent(bin_index, det + 1, counts)
    return hist2d


def write_csv(path: Path, rows: list[dict[str, float]], fieldnames: list[str]) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(
    root: Any,
    histograms: dict[int, Any],
    points_by_detector: dict[int, list[dict[str, float]]],
    coefficients: dict[int, dict[str, float]],
    args: argparse.Namespace,
) -> None:
    all_points = [
        point
        for det in sorted(points_by_detector)
        for point in points_by_detector[det]
    ]
    write_csv(
        Path(args.calibration_points),
        all_points,
        [
            "detector",
            "peak_index",
            "centroid",
            "centroid_error",
            "sigma",
            "gamma_energy",
            "fit_low",
            "fit_high",
        ],
    )

    coefficient_rows = [
        {
            "detector": det,
            "intercept": coefficients[det]["intercept"],
            "slope": coefficients[det]["slope"],
            "n_peaks": coefficients[det]["n_peaks"],
        }
        for det in sorted(coefficients)
    ]
    write_csv(
        Path(args.coefficients),
        coefficient_rows,
        ["detector", "intercept", "slope", "n_peaks"],
    )

    if args.cal_range:
        cal_range = tuple(args.cal_range)
    else:
        cal_range = calibrated_hist_range(histograms, coefficients)

    calibrated_hists = {}
    summed = root.TH1D(
        "labr_energy_cal_sum",
        "Summed calibrated LaBr energy;Energy;Counts",
        args.cal_bins,
        cal_range[0],
        cal_range[1],
    )
    for det in sorted(coefficients):
        hist = calibrate_histogram(
            root,
            det,
            histograms[det],
            coefficients[det],
            args.cal_bins,
            cal_range,
        )
        calibrated_hists[det] = hist
        summed.Add(hist)
    detector_index_hist = make_detector_index_histogram(
        root,
        "labr_energy_cal_vs_detector_index",
        "Calibrated LaBr energy vs detector index;Energy;Detector index",
        calibrated_hists,
        args.cal_bins,
        cal_range,
    )

    output_file = root.TFile(str(args.output), "RECREATE")
    for hist in calibrated_hists.values():
        hist.Write()
    summed.Write()
    detector_index_hist.Write()

    canvas = root.TCanvas("calibrated_labr_sum_canvas", "calibrated_labr_sum_canvas", 1000, 700)
    summed.Draw()
    canvas.Write()
    detector_canvas = root.TCanvas(
        "calibrated_labr_detector_index_canvas",
        "calibrated_labr_detector_index_canvas",
        1000,
        700,
    )
    detector_index_hist.Draw("COLZ")
    detector_canvas.Write()
    for det, hist in calibrated_hists.items():
        det_canvas = root.TCanvas(
            f"calibrated_labr_det{det}_canvas",
            f"calibrated_labr_det{det}_canvas",
            1000,
            700,
        )
        hist.Draw()
        det_canvas.Write()
    output_file.Close()

    if args.png_dir:
        args.png_dir.mkdir(parents=True, exist_ok=True)
        canvas.SaveAs(str(args.png_dir / "labr_energy_cal_sum.png"))
        detector_canvas.SaveAs(str(args.png_dir / "labr_energy_cal_vs_detector_index.png"))
        for det, hist in calibrated_hists.items():
            det_canvas = root.TCanvas(
                f"png_calibrated_labr_det{det}_canvas",
                f"png_calibrated_labr_det{det}_canvas",
                1000,
                700,
            )
            hist.Draw()
            det_canvas.SaveAs(str(args.png_dir / f"labr_energy_cal_det{det}.png"))

    print(f"Wrote {args.calibration_points}")
    print(f"Wrote {args.coefficients}")
    print(f"Wrote {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactively calibrate LaBr raw-energy ROOT histograms."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="ROOT file from draw_labr_histograms.py, for example labr_histograms.root.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Output ROOT file with calibrated detector and summed histograms. "
            "Default includes the input run label."
        ),
    )
    parser.add_argument(
        "--calibration-points",
        help="CSV output with fitted centroids and gamma energies. Default includes the input run label.",
    )
    parser.add_argument(
        "--coefficients",
        help="CSV output with linear calibration coefficients. Default includes the input run label.",
    )
    parser.add_argument(
        "--detectors",
        default="0-17",
        help="Detector list, for example 0-17 or 0,1,4.",
    )
    parser.add_argument(
        "--gamma-energies",
        help=(
            "Comma-separated gamma energies to reuse for every detector, "
            "for example 661.7,1173.2,1332.5. If omitted, you enter them once interactively."
        ),
    )
    parser.add_argument(
        "--fit-half-width",
        type=float,
        default=150.0,
        help="Half-width around clicked peak for Gaussian+linear fit.",
    )
    parser.add_argument("--cal-bins", type=int, default=4000)
    parser.add_argument(
        "--cal-range",
        nargs=2,
        type=float,
        metavar=("LOW", "HIGH"),
        help="Manual calibrated energy output range.",
    )
    parser.add_argument("--png-dir", type=Path, help="Optional PNG output directory.")
    parser.add_argument("--logy", action="store_true", help="Use log y-axis while fitting.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_default_output_paths(args)
    root = import_root()
    install_click_handler(root)
    detectors = parse_detectors(args.detectors)
    histograms = load_histograms(root, args.input, detectors)
    if not histograms:
        raise ValueError("No detector histograms were loaded")
    gamma_energies = get_gamma_energies(args)
    print("Using gamma energies: " + ", ".join(f"{energy:g}" for energy in gamma_energies))

    canvas = root.TCanvas("labr_calibration_canvas", "LaBr calibration", 1100, 800)
    if args.logy:
        canvas.SetLogy()

    points_by_detector: dict[int, list[dict[str, float]]] = {}
    coefficients: dict[int, dict[str, float]] = {}
    for det in detectors:
        if det not in histograms:
            continue
        hist = histograms[det]
        canvas.cd()
        hist.Draw()
        canvas.Update()
        if not yes_no(f"\nDetector {det}: do you want to calibrate this detector?", True):
            print(f"Detector {det}: skipped")
            continue

        points = collect_detector_calibration(
            root,
            canvas,
            det,
            hist,
            args.fit_half_width,
            gamma_energies,
        )
        if not points:
            print(f"Detector {det}: no accepted peaks; skipped")
            continue
        if len(points) < 2:
            print(
                f"Detector {det}: only one peak accepted; using intercept=0 and "
                "slope=gamma/centroid."
            )
        coeff = fit_linear_calibration(points)
        print(
            "Detector {} calibration: E_cal = {:.8g} + {:.8g} * E_raw".format(
                det, coeff["intercept"], coeff["slope"]
            )
        )
        points_by_detector[det] = points
        coefficients[det] = coeff

    if not coefficients:
        raise ValueError("No detectors were calibrated")

    write_outputs(root, histograms, points_by_detector, coefficients, args)


if __name__ == "__main__":
    main()
