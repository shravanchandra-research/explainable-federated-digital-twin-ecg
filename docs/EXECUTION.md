# Execution guide

## 1. Environment

Use an isolated Python environment.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
$env:PTBXL_ROOT = "C:\PTB-XL"
```

### Linux/macOS

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
export PTBXL_ROOT="/path/to/PTB-XL"
```

Download PTB-XL v1.0.3 from [PhysioNet](https://physionet.org/content/ptb-xl/1.0.3/) and extract it under `PTBXL_ROOT`. The configured directory must contain `ptbxl_database.csv`, `scp_statements.csv`, `RECORDS`, and `records100/`. Raw dataset files are not part of this repository.

## 2. Dataset inspection and initial analysis

Run commands from the repository root:

```powershell
python scripts/01_dataset_validation/experiment_1a_ptbxl_verify.py
python scripts/01_dataset_validation/experiment_1b_ptbxl_metadata.py
python scripts/02_diagnostic_analysis/experiment_2a_label_analysis.py
python scripts/02_diagnostic_analysis/experiment_2b_superclass_analysis.py
python scripts/02_diagnostic_analysis/experiment_2c_patient_fold_analysis.py
```

## 3. Initial preprocessing

```powershell
python scripts/03_initial_preprocessing/experiment_3a_preprocessing.py
python scripts/03_initial_preprocessing/experiment_3b_label_construction.py
python scripts/03_initial_preprocessing/experiment_3c_dataset_loader.py
```

## 4. Model-ready data pipeline

These scripts are staged and depend on source records and outputs from prior steps. Run in order:

```powershell
python scripts/04_model_ready_pipeline/experiment_4a_preprocessing.py
python scripts/04_model_ready_pipeline/experiment_4b_batch_preprocessing.py
python scripts/04_model_ready_pipeline/experiment_4c_preprocessed_dataset.py
python scripts/04_model_ready_pipeline/experiment_4d_label_alignment.py
python scripts/04_model_ready_pipeline/experiment_4e_model_ready_split.py
python scripts/04_model_ready_pipeline/experiment_4f_split_integrity.py
python scripts/04_model_ready_pipeline/experiment_4g_xai_input_validation.py
```

Experiment 4G is an **XAI input-integrity validator**; it does not generate explanations or compute explanation faithfulness/stability.

## 5. Full dataset preparation and HDF5 conversion

```powershell
python scripts/04_model_ready_pipeline/experiment_4h_full_preprocessing.py
python scripts/05_hdf5_preparation/experiment_5a_prepare_hdf5.py
```

These stages require the complete local PTB-XL 100 Hz waveform collection and adequate local disk space.

## 6. Syntax test

```bash
python -m unittest discover -s tests
```

This checks Python syntax for all deposited experiment scripts. It does not execute experiments against PTB-XL.

## 7. Outputs and caveats

Scripts write reports, arrays, indexes, and HDF5 artifacts under the local PTB-XL root, generally in its `experiments/` directory. Generated patient-level data are not automatically copied to GitHub.

The deposited source is a research-stage pipeline, not a single end-to-end command. Review each script's prerequisites and inspect generated indexes/error reports before using data for model training. The requirements file is not a historical lockfile; only WFDB 4.3.1 is pinned based on the version recorded in the workflow.
