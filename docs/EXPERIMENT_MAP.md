# Experiment map and deposit status

This inventory reflects the source files and logs available at the time of this repository update. A script being present does not by itself establish that it has been rerun in a clean environment.

| Stage | Deposited source | Purpose | Current evidence/status |
|---|---|---|---|
| 1A | `experiment_1a_ptbxl_verify.py` | Verify a 12-lead, 100 Hz PTB-XL waveform | Supplied validation log reports PASS for record 01000_lr |
| 1B | `experiment_1b_ptbxl_metadata.py` | Inspect record metadata and SCP codes | Supplied validation log reports PASS |
| 2A | `experiment_2a_label_analysis.py` | Diagnostic SCP-code analysis | Supplied text report included |
| 2B | `experiment_2b_superclass_analysis.py` | Map diagnostic codes to NORM, MI, STTC, CD, HYP | Supplied text report included |
| 2C | `experiment_2c_patient_fold_analysis.py` | Patient/fold integrity analysis | Supplied report says no patient crosses folds |
| 3A | `experiment_3a_preprocessing.py` | Single-record centering and normalization check | Supplied validation log reports PASS |
| 3B | `experiment_3b_label_construction.py` | Construct five-dimensional multi-label targets | Supplied validation log reports PASS |
| 3C | `experiment_3c_dataset_loader.py` | Validate loader, labels, and patient-preserving folds | Supplied validation log reports PASS |
| 4A | `experiment_4a_preprocessing.py` | Signal preprocessing/normalization validation | Source included; no execution log supplied here |
| 4B | `experiment_4b_batch_preprocessing.py` | Batch preprocessing validation | Source included; no execution log supplied here |
| 4C | `experiment_4c_preprocessed_dataset.py` | Generate a preprocessed sample dataset | Source included; no execution log supplied here |
| 4D | `experiment_4d_label_alignment.py` | Align labels with preprocessed data | Source included; no execution log supplied here |
| 4E | `experiment_4e_model_ready_split.py` | Build model-ready split/index | Source included; no execution log supplied here |
| 4F | `experiment_4f_split_integrity.py` | Check class distribution and split overlap | Source included; no execution log supplied here |
| 4G | `experiment_4g_xai_input_validation.py` | Validate XAI input files, shapes, finite values, and labels | Input validation only; not an explanation algorithm |
| 4H | `experiment_4h_full_preprocessing.py` | Full PTB-XL preprocessing and model-ready index creation | Source included; requires the full local 100 Hz dataset |
| 5A | `experiment_5a_prepare_hdf5.py` | Convert prepared arrays/index into HDF5 | Source included; requires successful 4H output |

## Not yet deposited

The available project files do not contain source implementations for:

1. Centralized 1D CNN training/evaluation and its validated test predictions.
2. FedAvg/FedAdam training and client simulation code.
3. Patient-level adaptation/personalization and associated prediction-generation code.
4. Integrated Gradients or another actual explanation-generation method, plus faithfulness/stability evaluation.
5. Complete runtime/energy profiling scripts and the environment/hardware details needed to reproduce the estimates.

The four federated JSON files are retained as reported result artifacts. They are not substitutes for the missing training code and prediction-level evidence.
