# ML Calibration Workflow

This repository contains the scripts used to build LaBr energy calibration
histograms, calibrate the LaBr energies, extract the timing-model dataframe,
train the 6D time-correction model, and apply that model to new ROOT runs.

The scripts assume the raw ROOT files contain branches like:

- `rootout/labr/labr.energy[18]`
- `rootout/labr/labr.time[18]`
- `rootout/NpspmtCeBr/NpspmtCeBr.dytime`

In commands below, the final positional number is a percentage of the run to
read. For example, `1` means `1%`, and `0.01` means `0.01%`. If you omit it,
the scripts read the full input.

## 1. Set Up The Environment

On Linux terminal:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements_extract_test_6D.txt
```

On the analysis server, make sure the ROOT environment is loaded before running
scripts that use PyROOT, especially the interactive calibration script.

## 2. Draw Raw LaBr Energy Histograms

Start by drawing raw `labr.energy` histograms from one ROOT file or multiple
ROOT files.

```bash
python draw_labr_histograms.py run-0989-forward-1-sorted.root \
  --output labr_histograms_run-0989.root \
  --energy-bin-width 10 \
  --png-dir labr_pngs
```

For multiple files or a glob:

```bash
python draw_labr_histograms.py "run*.root" \
  --output labr_histograms_runs.root \
  --energy-bin-width 10
```

Before calibration coefficients exist, use `--use-root-ecal` only if you want
to draw the ROOT file's existing `labr.ecal` branch. Otherwise, for the first
energy calibration, the important output is the raw `labr.energy` histograms.

Useful options:

- `--workers 4` controls parallel workers.
- `--energy-range LOW HIGH` skips raw-energy range scanning.
- `--ecal-range LOW HIGH` skips calibrated-energy range scanning.
- `--stop N` reads only the first `N` events.

## 3. Calibrate LaBr Energy

Use the raw-energy histograms to interactively fit peaks for each detector.

```bash
python calibrate_labr_energy_hists.py labr_histograms_run-0989.root \
  --fit-half-width 1500 \
  --logy
```

For each detector, the script asks whether to calibrate it. If yes, it asks how
many peaks to use. For each peak, zoom/pan in the ROOT canvas first, then click
near the peak. The script fits a single Gaussian plus linear background and asks
for the known gamma energy.

Default outputs include the run label:

- `labr_energy_calibration_points_run-0989.csv`
- `labr_energy_calibration_coefficients_run-0989.csv`
- `labr_energy_calibrated_run-0989.root`

The coefficient CSV contains the linear calibration for each detector:

```text
Ecal = intercept + slope * labr.energy
```

Only detectors listed in this CSV are used later when calibrated energies are
calculated from raw `labr.energy`.

## 4. Redraw LaBr Histograms With Calibration

After energy calibration, redraw raw and calibrated histograms using the
coefficient CSV.

```bash
python draw_labr_histograms.py run-0989-forward-1-sorted.root 1 \
  --coefficients labr_energy_calibration_coefficients_run-0989.csv \
  --output labr_histograms_calibrated_run-0989.root \
  --energy-bin-width 10 \
  --ecal-bin-width 1 \
  --png-dir labr_cal_pngs
```

This calculates `labr_ecal` from the coefficient CSV, not from ROOT's
`labr.ecal` branch. If you explicitly want the ROOT branch instead, add:

```bash
--use-root-ecal
```

## 5. Extract The 6D Timing Dataframe

Extract the dataframe used by `test_6D 1.py`. This reads the raw ROOT file,
calculates calibrated `Ei` and `Ej` from `labr.energy`, and keeps only detectors
that have coefficients.

```bash
python extract_test_6D_input.py run-0989-forward-1-sorted.root \
  --coefficients labr_energy_calibration_coefficients_run-0989.csv
```

Default outputs include the run label:

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

The timing model trains on `Ei, index_i, Ej, index_j` and predicts `tdiff`.

## 6. Train The 6D Time-Correction Model

Train the model using the extracted `.pkl` file.

```bash
python "test_6D 1.py" test_6D_input_run-0989.pkl
```

Default outputs include the input run label:

- `trained_model_run-0989.joblib`
- `test_6D_output_run-0989.root`

The model uses:

- independent variables: `Ei, index_i, Ej, index_j`
- target variable: `tdiff`
- training range: `tdiff` from `-20` to `20`
- correction definition: `tdiff_corrected = tdiff - tpred`

The ROOT output contains diagnostics such as:

- `Ej_Vs_T_Diff`
- `Ej_Vs_T_Diff_Corrected`
- `Prompt_Response`
- `Prompt_Response_Corrected`
- detector-pair timing maps

## 7. Apply Time Corrections To A New Run

For a new raw ROOT run, apply the same energy calibration first, then run the
trained timing model on calibrated energies.

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

The output tree is named `TimeCorrection` and contains:

- `file_index`
- `entry`
- `Ei`
- `index_i`
- `Ej`
- `index_j`
- `T_Diff`
- `T_pred`
- `T_Diff_Corrected`

The output also contains diagnostics similar to `test_6D`, plus 1D histograms
for LaBr[8] against every other detector:

- `Labr8_minus_Labr{i}_Before`
- `Labr8_minus_Labr{i}_After`

For fast histogram-only output, skip the large pair-by-pair tree:

```bash
python apply_time_corrections.py run-0990-forward-1-sorted.root \
  --model trained_model_run-0989.joblib \
  --coefficients labr_energy_calibration_coefficients_run-0989.csv \
  --time-range -20 20 \
  --corrected-time-range -20 20 \
  --energy-range 0 50000 \
  --no-tree
```

Use `--compression-level 0` for faster but larger output files.

## 8. Multiple Runs And Globs

Most scripts accept multiple files and glob patterns:

```bash
python extract_test_6D_input.py "run*.root" 1 \
  --coefficients labr_energy_calibration_coefficients_run-0989.csv
```

```bash
python apply_time_corrections.py "run*.root" 1 \
  --model trained_model_run-0989.joblib \
  --coefficients labr_energy_calibration_coefficients_run-0989.csv \
  --time-range -20 20 \
  --corrected-time-range -20 20 \
  --energy-range 0 50000 \
  --no-tree
```

When multiple input runs are used, the output name includes a combined run
label unless you pass `--output`.

## 9. Sync To Remote Server

To automatically sync code changes to the remote analysis folder:

```bash
./sync_to_remote.sh 2
```

This watches the local folder and syncs every 2 seconds to:

```text
gaballah@nsclgw1.nscl.msu.edu:/mnt/analysis/e23055/tg/ML/
```

The sync script intentionally excludes large/generated files such as `.root`,
`.joblib`, virtual environments, and extracted dataframe outputs.

## Notes

- Quote `"test_6D 1.py"` because the filename contains a space.
- If a script seems stuck after the progress bar reaches 100%, it may be
  writing or closing a large ROOT file.
- For timing histograms, keep `--time-range` and `--corrected-time-range`
  reasonably focused. With a `0.01` bin width, very wide ranges create huge
  ROOT histograms.
- If you only need diagnostic histograms from `apply_time_corrections.py`, use
  `--no-tree`.
