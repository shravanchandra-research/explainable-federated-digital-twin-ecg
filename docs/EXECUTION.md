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

The directory must contain `ptbxl_database.csv`, `scp_statements.csv`, `RECORDS` and the `records100/` waveform folder. The raw dataset is not redistributed in this repository.

## 2. Install dependencies

For the PTB-XL preparation scripts:
```bash
python -m pip install -r requirements.txt
```

For centralized CNN, federated learning, adaptation, explainability and resource profiling:
```bash
python -m pip install -r requirements-reference.txt
python -m pip install -e .
```

## 3. Run the PTB-XL experiment pipeline

Run from the repository root and preserve the order:
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

The resulting HDF5 used by the EFDT reference package is expected to expose `X (N,1000,12)`, `Y (N,5)`, `ecg_id`, `patient_id`, and `fold`. Inspect your local Experiment 5A output to confirm this schema before training.

## 4. Run model experiments

Centralized CNN:
```bash
python -m efdt centralized --h5 /path/to/ptbxl_100.h5 --seed 17
```

Federated IID FedAvg and FedAdam:
```bash
python -m efdt federated --h5 /path/to/ptbxl_100.h5 --optimizer fedavg --allocation iid --seed 17
python -m efdt federated --h5 /path/to/ptbxl_100.h5 --optimizer fedadam --allocation iid --seed 17
```

Federated non-IID:
```bash
python -m efdt federated --h5 /path/to/ptbxl_100.h5 --optimizer fedavg --allocation non-iid --alpha 0.3 --seed 17
python -m efdt federated --h5 /path/to/ptbxl_100.h5 --optimizer fedadam --allocation non-iid --alpha 0.3 --seed 17
```

Adaptation from a trained checkpoint:
```bash
python -m efdt adapt --h5 /path/to/ptbxl_100.h5 --checkpoint outputs/centralized/centralized_seed17.pt
```

Integrated Gradients and explanation audits:
```bash
python -m efdt explain --h5 /path/to/ptbxl_100.h5 --checkpoint outputs/centralized/centralized_seed17.pt --row 100 --target-class 1
```

Resource calculations; use `--measure` only to time your current host device:
```bash
python -m efdt resources --output outputs/resource_profile.json
python -m efdt resources --measure --device cpu --warmup 20 --repetitions 100
```

For every subcommand, use `python -m efdt SUBCOMMAND --help` to inspect options. Store each run's config, seed, checkpoint, client allocation and prediction outputs. Never overwrite the author-validated historical JSONs under `results/federated/`.

## 5. Run tests

```bash
python -m unittest discover -s tests -v
python -m efdt --help
```

The CI pipeline performs syntax compilation, reference unit tests and CLI startup. It does not download or rerun the complete PTB-XL clinical dataset.

## 6. Result and interpretation guidance

The original project-validated federated metrics and adaptation figure are stored in `results/`. New code execution writes newly generated outputs to `outputs/` by default. Keep those outputs separate from the project-validated historical values.

The reference FL implementation uses simulated, patient-disjoint clients; this is not a multi-hospital deployment. Patient adaptation requires repeated ECGs; query recordings must remain disjoint from support recordings. The Digital Twin code is retrospective state replay. Scenario latency/energy estimates based on assumed throughput and power are not physical hardware measurements.
