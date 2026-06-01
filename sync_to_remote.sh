#!/usr/bin/env bash
# Author: Tawfik Gaballah
# GitHub: tawfikgaballah
# Project: ML-LaBr3-Calibration

set -euo pipefail

REMOTE="gaballah@nsclgw1.nscl.msu.edu:/mnt/analysis/e23055/tg/ML/"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/"
INTERVAL_SECONDS="${1:-5}"
SSH_CONTROL_PATH="${HOME}/.ssh/cm-%r@%h:%p"
SSH_OPTS="-o ControlMaster=auto -o ControlPersist=10m -o ControlPath=${SSH_CONTROL_PATH}"

sync_once() {
  local delete_flag=()
  if [[ "${DELETE_REMOTE:-0}" == "1" ]]; then
    delete_flag=(--delete)
  fi

  rsync -az --progress -e "ssh ${SSH_OPTS}" "${delete_flag[@]}" --info=stats1,name1 \
    --exclude ".venv/" \
    --exclude ".venv_wsl/" \
    --exclude "__pycache__/" \
    --exclude "*.pyc" \
    --exclude "*.root" \
    --exclude "*.joblib" \
    --exclude "labr_pngs/" \
    --exclude "labr_cal_pngs/" \
    --exclude "ei_ecal_pngs/" \
    --exclude "applied_ei_ecal_pngs/" \
    --exclude "labr_energy_calibration_points.csv" \
    --exclude "labr_energy_calibration_coefficients.csv" \
    --exclude "test_6D_input*.csv" \
    --exclude "test_6D_input*.pkl" \
    --exclude "test_6D_input_smoke*" \
    --exclude "test_6D_input_cli_smoke*" \
    "${LOCAL_DIR}" "${REMOTE}"
}

snapshot() {
  find "${LOCAL_DIR}" \
    -path "${LOCAL_DIR}.venv" -prune -o \
    -path "${LOCAL_DIR}.venv_wsl" -prune -o \
    -path "${LOCAL_DIR}__pycache__" -prune -o \
    -path "${LOCAL_DIR}labr_pngs" -prune -o \
    -path "${LOCAL_DIR}labr_cal_pngs" -prune -o \
    -path "${LOCAL_DIR}ei_ecal_pngs" -prune -o \
    -path "${LOCAL_DIR}applied_ei_ecal_pngs" -prune -o \
    -name "*.root" -prune -o \
    -name "*.joblib" -prune -o \
    -type f -printf "%P %T@ %s\n" 2>/dev/null | sort
}

echo "Watching ${LOCAL_DIR}"
echo "Sync target: ${REMOTE}"
echo "Polling every ${INTERVAL_SECONDS}s. Press Ctrl+C to stop."

last_snapshot="$(snapshot)"
sync_once

while true; do
  sleep "${INTERVAL_SECONDS}"
  current_snapshot="$(snapshot)"
  if [[ "${current_snapshot}" != "${last_snapshot}" ]]; then
    echo "Changes detected at $(date '+%Y-%m-%d %H:%M:%S'); syncing..."
    sync_once
    last_snapshot="${current_snapshot}"
  fi
done
