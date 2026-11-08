# Execution guide

## 1. Prepare the environment

Use a supported Python installation and create an isolated virtual environment.

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

Download PTB-XL v1.0.3 from [PhysioNet](https://physionet.org/content/ptb-xl/1.0.3/) and extract it under the directory referenced by `PTBXL_ROOT`. That directory should contain `ptbxl_database.csv`, `scp_statements.csv`, `RECORDS`, and `records100/`.

Do not place the downloaded dataset under version control.

## 2. Run the experiments

Run scripts from the repository root. Example:

```powershell
python scripts/01_dataset_verification/experiment_1a_ptbxl_verify.py
python scripts/01_dataset_verification/experiment_1b_ptbxl_metadata.py
python scripts/02_diagnostic_analysis/experiment_2a_label_analysis.py
python scripts/02_diagnostic_analysis/experiment_2b_superclass_analysis.py
python scripts/02_diagnostic_analysis/experiment_2c_patient_fold_analysis.py
python scripts/03_initial_preprocessing/experiment_3a_preprocessing.py
python scripts/03_initial_preprocessing/experiment_3b_label_construction.py
python scripts/03_initial_preprocessing/experiment_3c_dataset_loader.py
```

Experiments 4A–4G are staged data-preparation and validation scripts. Run them in order after the required source records and prior-stage output files are present:

```powershell
python scripts/04_model_ready_pipeline/experiment_4a_preprocessing.py
python scripts/04_model_ready_pipeline/experiment_4b_batch_preprocessing.py
python scripts/04_model_ready_pipeline/experiment_4c_preprocessed_dataset.py
python scripts/04_model_ready_pipeline/experiment_4d_label_alignment.py
python scripts/04_model_ready_pipeline/experiment_4e_model_ready_split.py
python scripts/04_model_ready_pipeline/experiment_4f_split_integrity.py
python scripts/04_model_ready_pipeline/experiment_4g_xai_input_validation.py
```

For full-dataset preparation, run 4H and then 5A:

```powershell
python scripts/04_model_ready_pipeline/experiment_4h_full_preprocessing.py
python scripts/05_hdf5_preparation/experiment_5a_prepare_hdf5.py
```

## 3. Outputs

The scripts write working files and reports below the PTB-XL root directory, mainly in its `experiments/` subdirectory. These are local generated outputs and are not automatically uploaded to GitHub.

## 4. Important cautions

- The scripts are supplied research-stage scripts, not a single fully automated pipeline. Some stages depend on files created by earlier stages.
- Full-dataset preprocessing and HDF5 conversion require the complete PTB-XL v1.0.3 100 Hz waveform collection and adequate local disk space.
- The 4G script validates data inputs for subsequent XAI experiments; it does not calculate explanations.
- The deposited requirements file is not a historical lockfile. Exact versions beyond WFDB 4.3.1 have not been established from the available run records.
- Inspect generated indexes and error reports before training any model.
