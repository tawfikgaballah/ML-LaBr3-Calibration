# ML LaBr3 Calibration Workflow

This repository contains scripts for the LaBr3 energy-calibration and
time-walk-correction workflow:

1. Draw raw LaBr energy histograms from ROOT runs.
2. Interactively fit gamma peaks and create per-detector energy coefficients.
3. Redraw raw and calibrated LaBr histograms.
4. Extract the dataframe used by the 6D timing model.
5. Train the timing model on `Ei, index_i, Ej, index_j -> tdiff`.
6. Apply energy calibration and timing correction to new ROOT runs.

All workflow scripts include the project header:

```text
Author: Tawfik Gaballah
GitHub: tawfikgaballah
Project: ML-LaBr3-Calibration
```

The main raw ROOT branches used by the scripts are:

- `rootout/labr/labr.energy[18]`
- `rootout/labr/labr.ecal[18]`
- `rootout/labr/labr.time[18]`
- `rootout/NpspmtCeBr/NpspmtCeBr.dytime`

When a command accepts a final positional number, that number is a percentage
of the input run. For example, `1` means `1%`, and `0.01` means `0.01%`.

## Environment

On WSL or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements_extract_test_6D.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements_extract_test_6D.txt
```

On the analysis server, load the ROOT environment before running scripts that
need PyROOT, especially `calibrate_labr_energy_hists.py`.

## 1. Draw Raw LaBr Histograms

Use `draw_labr_histograms.py` to draw raw `labr.energy` histograms and, when
possible, calibrated-energy histograms.

```bash
python draw_labr_histograms.py run-0989-forward-1-sorted.root \
  --output labr_histograms_run-0989.root \
  --energy-bin-width 10 \
  --png-dir labr_pngs
```

For multiple files or a glob:

```bash
python draw_labr_histograms.py "run*.root" 1 \
  --output labr_histograms_runs.root \
  --energy-bin-width 10
```

Current behavior:

- Raw `labr.energy` bin width defaults to `10`.
- Calculated `labr_ecal` bin width defaults to `1`.
- Values must be `> 0` by default for both filling and automatic range scans.
- Use `--include-invalid` only if you intentionally want zero/negative
  placeholder values.
- Use `--workers N` to parallelize range scanning and filling. Use `--workers 1`
  to disable multiprocessing.
- Detector-index 2D histograms are written with energy on the x axis and
  detector index on the y axis from `0` to `18`.
- Automatic ranges are capped by `--max-ecal-bins` and `--max-energy-bins` to
  avoid huge arrays when ROOT contains bad outliers.

Before energy-calibration coefficients exist, the important output is the raw
`labr.energy` spectra. After coefficients exist, pass them explicitly:

```bash
python draw_labr_histograms.py run-0989-forward-1-sorted.root \
  --coefficients labr_energy_calibration_coefficients_run-0989.csv \
  --output labr_histograms_calibrated_run-0989.root \
  --energy-bin-width 10 \
  --ecal-bin-width 1 \
  --png-dir labr_cal_pngs
```

With coefficients, only detectors listed in the coefficient CSV are used for
calculated `labr_ecal`. If you want to ignore the CSV and draw the ROOT file's
own `labr.ecal` branch for all detectors, use:

```bash
--use-root-ecal
```

If ROOT `labr.ecal` has bad outliers, give manual ranges:

```bash
--energy-range 0 70000 --ecal-range 0 50000
```

## 2. Interactively Calibrate LaBr Energy

Use `calibrate_labr_energy_hists.py` on the histogram ROOT file from the
previous step.

```bash
python calibrate_labr_energy_hists.py labr_histograms_run-0989.root \
  --gamma-energies 661.7,1173.2,1332.5 \
  --fit-half-width 1500 \
  --logy
```

Current behavior:

- Gamma energies are entered once with `--gamma-energies` and reused for every
  detector.
- If `--gamma-energies` is omitted, the script asks for the gamma energies once
  at the beginning.
- For each detector, the script asks whether to calibrate it.
- You can zoom or pan in the ROOT canvas before clicking each peak.
- Each clicked peak is fit with a single Gaussian plus linear background.
- The old double-Gaussian fit path was removed.
- Empty histograms and 2D histograms are skipped.
- Output filenames include the input run label when possible.

Default outputs for `labr_histograms_run-0989.root` are:

- `labr_energy_calibration_points_run-0989.csv`
- `labr_energy_calibration_coefficients_run-0989.csv`
- `labr_energy_calibrated_run-0989.root`

The coefficient CSV defines:

```text
Ecal = intercept + slope * labr.energy
```

The calibrated ROOT output contains individual calibrated detector histograms,
the summed calibrated histogram, and a calibrated-energy versus detector-index
diagnostic.

## 3. Extract The 6D Timing Dataframe

Use `extract_test_6D_input.py` to read raw ROOT branches directly and build the
dataframes used by `test_6D 1.py`.

```bash
python extract_test_6D_input.py run-0989-forward-1-sorted.root \
  --coefficients labr_energy_calibration_coefficients_run-0989.csv
```

For multiple files or a glob:

```bash
python extract_test_6D_input.py "run*.root" 1 \
  --coefficients labr_energy_calibration_coefficients_run-0989.csv
```

Current behavior:

- `Ei` and `Ej` are calculated from raw `labr.energy` using the coefficient CSV.
- Only detectors with coefficients are used.
- Same-detector pairs are removed: `index_i != index_j`.
- The raw pair loop stores only `pair_tdiff_min < tdiff < pair_tdiff_max`,
  defaulting to `-1000` to `1000`.
- The saved `df_filtered` uses `--tdiff-min` and `--tdiff-max`, defaulting to
  `-100` to `100`.
- Progress is shown while extracting.
- Output filenames include the input run label.

Useful options:

```bash
--fraction 0.01
--min-energy 0
--min-ecal 0
--max-ecal 50000
--min-time 100
--min-dynode-time 100
--pair-tdiff-min -1000
--pair-tdiff-max 1000
```

The default outputs for run `0989` are:

- `test_6D_input_run-0989.pkl`
- `test_6D_input_run-0989_df.csv`
- `test_6D_input_run-0989_df_filtered.csv`

The dataframe columns are:

- `Ei`
- `index_i`
- `Ej`
- `index_j`
- `tdiff`
- `tdiff_aligned`

The current timing workflow trains on `tdiff`, not `tdiff_aligned`.

## 4. Train The 6D Timing Model

Use `test_6D 1.py` to train the timing model.

```bash
python "test_6D 1.py" test_6D_input_run-0989.pkl
```

Default outputs include the input run label:

- `trained_model_run-0989.joblib`
- `test_6D_output_run-0989.root`

Current model behavior:

- Inputs: `Ei`, `index_i`, `Ej`, `index_j`
- Target: `tdiff`
- Default training window: `-20 <= tdiff <= 20`
- Same-detector rows are removed: `index_i != index_j`
- The default held-out test fraction is `--test-size 0.1`
- `--data-fraction` samples the filtered dataset before the train/test split.
- ROOT diagnostics are drawn from the held-out test data only.
- Correction definition: `tdiff_corrected = tdiff - tpred`
- Prompt-response defaults are `--prompt-low 494.55` and `--prompt-high 527.45`.
- Time-axis bin width is `0.01`; energy-axis bin width is `10`.
- Progress messages and tqdm bars are shown for training, evaluating,
  predicting, and writing diagnostics.

Example using 20% of the filtered dataset, with 10% of that used for testing:

```bash
python "test_6D 1.py" test_6D_input_run-0989.pkl \
  --data-fraction 0.2 \
  --test-size 0.1
```

The model is a Keras dense neural network inside a scikit-learn pipeline. The
preprocessor applies `log1p` to energies, one-hot encodes detector indices, and
scales the energy columns. The loss is Huber loss.

The ROOT output contains:

- `TreeOutput`
- `Ej_Vs_T_Diff`
- `Ej_Vs_T_Diff_Corrected`
- `Prompt_Response`
- `Prompt_Response_Corrected`
- `T_Diff_Counts_Vs_Detector_Pair`
- `Average_T_Diff_Corrected_Vs_Detector_Pair`

## 5. Apply Timing Corrections To New Runs

Use `apply_time_corrections.py` to apply the energy calibration and timing
model to new raw ROOT runs.

```bash
python apply_time_corrections.py run-0990-forward-1-sorted.root \
  --model trained_model_run-0989.joblib \
  --coefficients labr_energy_calibration_coefficients_run-0989.csv \
  --time-range -20 20 \
  --corrected-time-range -20 20 \
  --energy-range 0 50000
```

Default outputs include the new input run label:

- `time_corrected_run-0990.root`
- `time_corrected_run-0990.files.txt`

Current behavior:

- Calibrated energies are calculated first from raw `labr.energy`.
- Only detectors with energy-calibration coefficients are considered.
- Pairs are restricted to detector indices known by the trained timing model.
- Same-detector pairs are removed: `index_i != index_j`.
- Progress bars are shown for range scanning and output writing.
- The output tree is named `TimeCorrection`.
- The output includes the same main diagnostics as `test_6D 1.py`.
- It also writes 1D before/after histograms for `Labr[reference] - Labr[i]`.
- The default reference detector is `8`.

The output tree contains:

- `file_index`
- `entry`
- `Ei`
- `index_i`
- `Ej`
- `index_j`
- `T_Diff`
- `T_pred`
- `T_Diff_Corrected`

For faster histogram-only output, skip the large pair-by-pair tree:

```bash
python apply_time_corrections.py run-0990-forward-1-sorted.root \
  --model trained_model_run-0989.joblib \
  --coefficients labr_energy_calibration_coefficients_run-0989.csv \
  --time-range -20 20 \
  --corrected-time-range -20 20 \
  --energy-range 0 50000 \
  --no-tree
```

Use `--compression-level 0` for faster but larger ROOT output.

If the apply script skips a detector that has energy coefficients, check the
model message:

```text
Restricting pairs to detector indices known by the model: ...
```

That means the timing model was trained without that detector index. Retrain
the timing model with extracted data that includes that detector if you want
detector-specific timing corrections for it.

## 6. Optional Ei To Ecal ML Utilities

The primary energy calibration path is the interactive linear coefficient CSV.
There are also optional ML utilities for studying energy calibration.

Train `Ecal` as a function of `Ei` and `index_i` using extracted `test_6D`
input plus the coefficient CSV:

```bash
python train_ei_ecal_model.py test_6D_input_run-0989.pkl \
  --coefficients labr_energy_calibration_coefficients_run-0989.csv \
  --model-output ei_to_ecal_model.joblib \
  --hist-output ei_ecal_histograms.root
```

This writes raw/calibrated diagnostic histograms with raw bin width `10` and
calibrated bin width `1`, plus detector-index versus energy diagnostics.

Apply that model to another ROOT run:

```bash
python apply_ei_ecal_model.py run-0990-forward-1-sorted.root \
  --model ei_to_ecal_model.joblib
```

There is also a clover-trained utility:

```bash
python train_clover_apply_labr_calibration.py run-0989-forward-1-sorted.root \
  --model-output clover_energy_to_ecal_model.joblib \
  --predictions-output labr_ecal_from_clover_model.root
```

It trains on `clover.energy -> clover.ecal` and applies the learned calibration
function to `labr.energy`.


## Practical Notes

- Quote `"test_6D 1.py"` because the filename contains a space.
- Most scripts accept multiple files and glob patterns such as `"run*.root"`.
- For timing histograms, keep `--time-range` and `--corrected-time-range`
  reasonably focused. With `0.01` time bins, very wide ranges create large ROOT
  histograms.
- If a script reaches a progress bar at 100% and appears quiet, it may still be
  writing or closing a large ROOT file.
- Use `--no-tree` in `apply_time_corrections.py` when you only need diagnostic
  histograms.
- If ROOT `labr.ecal` contains very large bad values, use manual `--ecal-range`
  or prefer coefficient-based calculated `labr_ecal`.
