# Results archive

This directory contains the federated result artifacts and aggregate validation logs currently deposited for the EFDT research project. Reserved subfolders identify where additional manuscript evidence should be added during revision.

## Federated learning results

- `federated/FedAvg/metrics_IID.json`
- `federated/FedAvg/metrics_non-IID.json`
- `federated/FedAdam/metrics_IID.json`
- `federated/FedAdam/metrics_non-IID.json`
- `federated/federated_learning_consolidated_results.csv`: supplied consolidated summary
- `federated/federated_metrics_exact.csv`: summary with client F1 population standard deviations recomputed from each ten-client JSON vector

The project owner confirms the four JSON results are validated project outputs. They are retained unchanged. New outputs from `src/efdt/` must be saved separately and must not overwrite these historical artifacts.

## Other reserved evidence folders

- `centralized/`: original centralized baseline configuration, predictions, metrics, logs and checkpoint provenance. The manuscript reports Macro-F1 0.901240 and Macro-AUROC 0.991450; the corresponding centralized result files are not yet deposited here.
- `adaptation/`: sanitized paired patient-level results and analysis script. Project records document mean MI F1 change −0.076190; the patient-level CSV and class-wise plot are not currently in this Git tree.
- `explainability/`: attribution outputs and quantitative explanation audits, if run.
- `resource/`: model complexity and resource calculation outputs with assumptions.

## Dataset validation

`validation_logs/` contains aggregate reports for Experiments 1A–3C. The public Experiment 1B text file is sanitized; record-level patient identifiers and demographic fields are not included.

## Figures

The `figures/` directory currently contains submission guidance only. Add the approved framework diagram, preprocessing illustrations and result plots with source scripts and captions during revision. Clearly distinguish measured, simulated and scenario-derived values.

## Data safety

Raw PTB-XL waveforms and unnecessary patient-level metadata are not included. Keep newly generated predictions, model checkpoints and patient-adaptation records on controlled local storage. Public result artifacts should be sanitized and checked for re-identification risk.
