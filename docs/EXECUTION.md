# Execution guide

## 1. Prepare the PTB-XL dataset

Download PTB-XL v1.0.3 from [PhysioNet](https://physionet.org/content/ptb-xl/1.0.3/), extract it locally and set `PTBXL_ROOT`.

### Windows PowerShell
```powershell
$env:PTBXL_ROOT = "C:\PTB-XL"
```

### Linux/macOS
```bash
export PTBXL_ROOT="/path/to/PTB-XL"
```

The directory must contain `ptbxl_database.csv`, `scp_statements.csv`, `RECORDS` and `records100/`. Raw PTB-XL data are not redistributed.

## 2. Install dependencies

For the dataset preparation scripts:
```bash
python -m pip install -r requirements.txt
```

For centralized CNN, federated learning, adaptation, explainability and resource profiling:
```bash
python -m pip install -r requirements-reference.txt
python -m pip install -e .
```

## 3. Run the PTB-XL preparation pipeline

Run from the repository root in the listed order:
```bash
python scripts/01_dataset_validation/experiment_1a_ptbxl_verify.py
python scripts/01_dataset_validation/experiment_1b_ptbxl_metadata.py
python scripts/02_diagnostic_analysis/experiment_2a_label_analysis.py
python scripts/02_diagnostic_analysis/experiment_2b_superclass_analysis.py
python scripts/02_diagnostic_analysis/experiment_2c_patient_fold_analysis.py
python scripts/03_initial_preprocessing/experiment_3a_preprocessing.py
python scripts/03_initial_preprocessing/experiment_3b_label_construction.py
python scripts/03_initial_preprocessing/experiment_3c_dataset_loader.py
python scripts/04_model_ready_pipeline/experiment_4a_preprocessing.py
python scripts/04_model_ready_pipeline/experiment_4b_batch_preprocessing.py
python scripts/04_model_ready_pipeline/experiment_4c_preprocessed_dataset.py
python scripts/04_model_ready_pipeline/experiment_4d_label_alignment.py
python scripts/04_model_ready_pipeline/experiment_4e_model_ready_split.py
python scripts/04_model_ready_pipeline/experiment_4f_split_integrity.py
python scripts/04_model_ready_pipeline/experiment_4g_xai_input_validation.py
python scripts/04_model_ready_pipeline/experiment_4h_full_preprocessing.py
python scripts/05_hdf5_preparation/experiment_5a_prepare_hdf5.py
```

The resulting Experiment 5A HDF5 uses `X (N,1000,12)`, `Y (N,5)`, `ecg_id`, `patient_id`, `fold` and `split` datasets. Check the file before launching model runs.

## 4. Run the model workflows

Centralized CNN:
```bash
python -m efdt centralized --h5 /path/to/ptbxl_4h_model_ready.h5 --seed 17
```

Federated IID:
```bash
python -m efdt federated --h5 /path/to/ptbxl_4h_model_ready.h5 --optimizer fedavg --allocation iid --seed 17
python -m efdt federated --h5 /path/to/ptbxl_4h_model_ready.h5 --optimizer fedadam --allocation iid --seed 17
```

Federated non-IID:
```bash
python -m efdt federated --h5 /path/to/ptbxl_4h_model_ready.h5 --optimizer fedavg --allocation non-iid --alpha 0.3 --seed 17
python -m efdt federated --h5 /path/to/ptbxl_4h_model_ready.h5 --optimizer fedadam --allocation non-iid --alpha 0.3 --seed 17
```

Patient adaptation:
```bash
python -m efdt adapt --h5 /path/to/ptbxl_4h_model_ready.h5 --checkpoint outputs/centralized/centralized_seed17.pt
```

Integrated Gradients and explanation audits:
```bash
python -m efdt explain --h5 /path/to/ptbxl_4h_model_ready.h5 --checkpoint outputs/centralized/centralized_seed17.pt --row 100 --target-class 1
```

Resource analysis:
```bash
python -m efdt resources --output outputs/resource_profile.json
python -m efdt resources --measure --device cpu --warmup 20 --repetitions 100
```

Use `python -m efdt SUBCOMMAND --help` to inspect parameters. The reference protocol is specified in `configs/reference_protocol.json`.

## 5. Test the complete package

```bash
python -m unittest discover -s tests -v
python -m efdt --help
```

GitHub Actions installs both dependency sets, runs reference unit tests, compiles all 17 experiment scripts and package modules, and checks CLI startup.

## 6. Result handling

The project-validated historic artifacts are preserved in `results/`. New runs write to `outputs/` by default. Do not overwrite validated JSON/CSV files with newly generated values. Include configuration, seed, dataset version, split, checkpoint, client allocation and prediction outputs for every new run.

The FL implementation uses patient-disjoint simulated clients, not real hospitals. Patient adaptation requires repeated ECGs and keeps support/query rows disjoint. The Digital Twin module is retrospective patient-state replay. Analytical resource values under assumed throughput and power are not physical hardware measurements.
