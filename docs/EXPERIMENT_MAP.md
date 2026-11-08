# Experiment and implementation map

This repository combines the PTB-XL processing experiments, the project-validated result artifacts and the executable model/reference modules.

| Stage | Source | Purpose | Result/evidence |
|---|---|---|---|
| 1A | `scripts/01_dataset_validation/experiment_1a_ptbxl_verify.py` | 12-lead waveform, 100 Hz, 1,000 samples and signal integrity | Validation log |
| 1B | `scripts/01_dataset_validation/experiment_1b_ptbxl_metadata.py` | Metadata and SCP-code inspection | Sanitized aggregate log |
| 2A–2C | `scripts/02_diagnostic_analysis/` | SCP counts, five superclass mapping, patient/fold integrity | Validation logs |
| 3A–3C | `scripts/03_initial_preprocessing/` | Centering/normalization, target construction and loader checks | Validation logs |
| 4A–4H | `scripts/04_model_ready_pipeline/` | Full preprocessing, batch preparation, label alignment, split checks and XAI input integrity | Local PTB-XL experiment outputs |
| 5A | `scripts/05_hdf5_preparation/experiment_5a_prepare_hdf5.py` | Prepare the model-ready HDF5 | Local Experiment 5A HDF5 output |
| Centralized CNN | `src/efdt/model.py`, `src/efdt/training.py` | Compact CNN, AdamW training, early stopping, validation-only thresholds | New run outputs |
| Federated learning | `src/efdt/federated.py` | Patient-disjoint IID/non-IID client simulation, FedAvg and FedAdam | Validated supplied result artifacts under `results/federated/`; new runs separately stored |
| Patient adaptation | `src/efdt/adaptation.py` | Support/query adaptation on eligible held-out patients | Supplied adaptation figure and mean MI delta |
| Digital Twin replay | `src/efdt/twin.py` | Retrospective patient-state updates | State JSON when run |
| Explainability | `src/efdt/explainability.py` | Integrated Gradients, perturbation faithfulness and stability utilities | Attribution arrays and JSON when run |
| Resource profile | `src/efdt/resource.py` | Parameter/MAC/memory calculation and host inference timing | Analytical scenario and optional host timing |

## Result provenance and interpretation

The supplied JSONs, CSV and figures are the project-validated outputs retained from the consolidated research archive. New outputs produced by `src/efdt/` are labelled `new_reference_implementation` and must be recorded separately. The Experiment 4G script is XAI input integrity validation; attribution generation and quantitative attribution utilities are in `src/efdt/explainability.py`.

The centralized baseline is executable; the supplied result archive does not contain a validated centralized Macro-F1 or Macro-AUROC value, so none is inferred from the federated results. Analytical resource estimates must be distinguished from physical measurements.
