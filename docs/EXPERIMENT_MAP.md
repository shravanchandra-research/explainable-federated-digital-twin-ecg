# Experiment map and deposit status

The repository contains source scripts for Experiments 1A–5A. All 17 scripts were syntax-compiled after changing their dataset root to read from `PTBXL_ROOT` (with the original `C:\PTB-XL` location retained as a fallback). Syntax compilation is not an end-to-end experiment rerun.

| Stage | Repository script | Purpose | Evidence/status |
|---|---|---|---|
| 1A | `scripts/01_dataset_validation/experiment_1a_ptbxl_verify.py` | Verify a 12-lead, 100 Hz waveform | Supplied log reports PASS for record 01000_lr |
| 1B | `scripts/01_dataset_validation/experiment_1b_ptbxl_metadata.py` | Inspect metadata and SCP codes | Sanitized aggregate summary included; record-level identifiers and demographics excluded |
| 2A | `scripts/02_diagnostic_analysis/experiment_2a_label_analysis.py` | Analyze diagnostic SCP codes | Supplied log is included |
| 2B | `scripts/02_diagnostic_analysis/experiment_2b_superclass_analysis.py` | Map codes to NORM, MI, STTC, CD, HYP | Supplied log is included |
| 2C | `scripts/02_diagnostic_analysis/experiment_2c_patient_fold_analysis.py` | Analyze patient/fold integrity | Supplied log reports no patient crossing folds |
| 3A | `scripts/03_initial_preprocessing/experiment_3a_preprocessing.py` | Initial signal preprocessing | Supplied log is included |
| 3B | `scripts/03_initial_preprocessing/experiment_3b_label_construction.py` | Construct five-superclass targets | Supplied log is included |
| 3C | `scripts/03_initial_preprocessing/experiment_3c_dataset_loader.py` | Validate loader, labels and folds | Supplied log is included |
| 4A | `scripts/04_model_ready_pipeline/experiment_4a_preprocessing.py` | Signal preprocessing/normalization | Source included; no supplied log in this deposit |
| 4B | `scripts/04_model_ready_pipeline/experiment_4b_batch_preprocessing.py` | Batch preprocessing | Source included; no supplied log in this deposit |
| 4C | `scripts/04_model_ready_pipeline/experiment_4c_preprocessed_dataset.py` | Generate preprocessed sample dataset | Source included; no supplied log |
| 4D | `scripts/04_model_ready_pipeline/experiment_4d_label_alignment.py` | Align labels and preprocessed records | Source included; no supplied log |
| 4E | `scripts/04_model_ready_pipeline/experiment_4e_model_ready_split.py` | Build model-ready index/split | Source included; no supplied log |
| 4F | `scripts/04_model_ready_pipeline/experiment_4f_split_integrity.py` | Check class distribution and split overlap | Source included; no supplied log |
| 4G | `scripts/04_model_ready_pipeline/experiment_4g_xai_input_validation.py` | Validate XAI input files, shapes, finite values and labels | Input validation only; not explanation generation |
| 4H | `scripts/04_model_ready_pipeline/experiment_4h_full_preprocessing.py` | Full PTB-XL preprocessing and model-ready index | Source included; requires full local 100 Hz dataset |
| 5A | `scripts/05_hdf5_preparation/experiment_5a_prepare_hdf5.py` | Convert prepared arrays/index to HDF5 | Source included; depends on successful 4H output |

## Not yet deposited

The available files do not include source implementations for:

1. Centralized 1D CNN training/evaluation and validated test predictions.
2. FedAvg/FedAdam training and client-simulation implementation.
3. Patient-level adaptation/personalization and prediction-generation code.
4. Integrated Gradients or another explanation-generation method, plus explanation faithfulness/stability evaluation.
5. Complete runtime/energy profiling scripts and hardware measurements.

The four federated JSON files are retained as supplied result artifacts, not as a substitute for missing training code or prediction-level evidence.
