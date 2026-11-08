# Results archive

This directory contains the project-validated result artifacts carried forward from the consolidated EFDT research archive.

## Federated learning results

- `federated/FedAvg/metrics_IID.json`
- `federated/FedAvg/metrics_non-IID.json`
- `federated/FedAdam/metrics_IID.json`
- `federated/FedAdam/metrics_non-IID.json`
- `federated/federated_learning_consolidated_results.csv`: supplied consolidated summary
- `federated/federated_metrics_exact.csv`: summary with client F1 population standard deviations recomputed from each ten-client JSON vector

The project owner confirms the four JSON results are validated project outputs. They are retained unchanged. New outputs from `src/efdt/` are saved separately under a user-selected output directory; they must not overwrite validated historical results.

## Patient adaptation

`patient_adaptation/personalization_deltas_boxplot.png` is the supplied class-wise F1-delta visualization. The reported mean MI F1 change is −0.076190. Do not invent unavailable per-class sample counts.

## Dataset validation

`validation_logs/` contains aggregate reports for Experiments 1A–3C. The public Experiment 1B text file is sanitized; record-level patient identifiers and demographic fields are not included.

## Figures

The complete ZIP package includes the corrected conceptual framework diagram, 12-lead ECG plot, preprocessing illustration, federated comparison plot and patient-adaptation figure under `figures/` and `results/patient_adaptation/`.

## Data safety

Raw PTB-XL waveforms and unnecessary patient-level metadata are not included. Keep newly generated predictions, model checkpoints and patient-adaptation records on controlled local storage.
