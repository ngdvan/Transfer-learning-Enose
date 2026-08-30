# Result reconstruction package

This directory contains the source datasets and code required to reproduce the numerical results reported in the manuscript. Trained model files are intentionally excluded. Models are created when the scripts run.

## Dataset directories

- `datasets/NH3_H2_dataset`: private NH3/H2 dataset.
- `datasets/dynamic_gas_mixture`: public dynamic gas mixture source and processed sequence cache - DOI: 10.24432/C5WP4C.
- `datasets/sensor_drift_dataset`: public sensor drift dataset batches 1-10 - DOI: 10.24432/C5MK6M.

`dataset_manifest.json` records file sizes and SHA-256 checksums.

## Environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Main experiments

Run commands from this `bin` directory.

```powershell
python code/same_device_session_transfer.py
python code/co_ethylene_session_transfer.py
python code/sensor_drift_transfer.py
```

The scripts reconstruct models during execution and write numerical outputs under `results/`. The code does not require packaged trained weights.

## Validation and ablation analyses

```powershell
python code/advanced_validation.py --help
```
