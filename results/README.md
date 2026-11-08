# Results archive

This directory contains the project-validated results carried forward from the consolidated EFDT research archive.

## Federated learning results

- `federated/FedAvg/metrics_IID.json`
- `federated/FedAvg/metrics_non-IID.json`
- `federated/FedAdam/metrics_IID.json`
- `federated/FedAdam/metrics_non-IID.json`
- `federated/federated_learning_consolidated_results.csv`: supplied consolidated summary
- `federated/federated_metrics_exact.csv`: summary with client F1 population standard deviations recomputed from each ten-client JSON vector

The four JSON results are treated as author-validated project outputs, per project confirmation. They are retained unchanged. New outputs from `src/efdt/` are stored separately under a user-selected output directory and must not overwrite these historical results.

## Patient adaptation

`patient_adaptation/personalization_deltas_boxplot.png` is the supplied class-wise F1-delta visualization. The reported mean MI F1 change is −0.076190. Interpret class-wise results descriptively; do not infer unavailable per-class sample counts.

## Dataset validation

`validation_logs/` holds aggregate reports for Experiments 1A–3C. The public Experiment 1B text file is sanitized; record-level patient identifiers and demographic fields are not included.

## Figures

The ZIP research package includes the corrected conceptual framework drawing, 12-lead ECG plot, preprocessing illustration and federated comparison plot under `figures/`. These are project figures; use the source figure files and figure notes for manuscript attribution.

## Data safety

Raw PTB-XL waveforms and unnecessary patient-level metadata are not included. Keep new model predictions, checkpoints and patient adaptation records on controlled local storage.
