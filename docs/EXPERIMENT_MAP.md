# Experiment and implementation map

This repository combines the PTB-XL processing experiments, supplied federated result artifacts and the executable model/reference modules.

| Stage | Source | Purpose | Result/evidence currently deposited |
|---|---|---|---|
| 1A | `scripts/01_dataset_validation/experiment_1a_ptbxl_verify.py` | 12-lead waveform, 100 Hz, 1,000 samples and signal integrity | Validation log |
| 1B | `scripts/01_dataset_validation/experiment_1b_ptbxl_metadata.py` | Metadata and SCP-code inspection | Sanitized aggregate log |
| 2A–2C | `scripts/02_diagnostic_analysis/` | SCP counts, five superclass mapping, patient/fold integrity | Validation logs |
| 3A–3C | `scripts/03_initial_preprocessing/` | Centering/normalization, target construction and loader checks | Validation logs |
| 4A–4H | `scripts/04_model_ready_pipeline/` | Full preprocessing, batch preparation, label alignment, split checks and XAI input integrity | Source scripts; generated local outputs are not committed |
| 5A | `scripts/05_hdf5_preparation/experiment_5a_prepare_hdf5.py` | Prepare the model-ready HDF5 | Source script; generated HDF5 is local and ignored |
| Centralized CNN | `src/efdt/model.py`, `src/efdt/training.py` | Compact CNN, AdamW training, early stopping, validation-only thresholds | Executable reference workflow; historical centralized result evidence reserved under `results/centralized/` |
| Federated learning | `src/efdt/federated.py` | Patient-disjoint IID/non-IID client simulation, FedAvg and FedAdam | Four supplied JSON artifacts and CSV summaries under `results/federated/` |
| Patient adaptation | `src/efdt/adaptation.py` | Support/query adaptation on eligible held-out patients | Reference implementation; mean MI ΔF1 is documented in project records; paired patient-level CSV/figure are not currently in this Git tree |
| Digital Twin replay | `src/efdt/twin.py` | Retrospective patient-state updates | Reference implementation and synthetic unit test |
| Explainability | `src/efdt/explainability.py` | Integrated Gradients, perturbation faithfulness and stability utilities | Reference implementation and synthetic unit test; no cohort-level results currently deposited |
| Resource profile | `src/efdt/resource.py` | Parameter/MAC/memory calculation and host inference timing | Reference implementation; deposit run outputs under `results/resource/` |

## Result provenance and interpretation

The four federated JSONs and two CSV summaries are present in the repository and are preserved as supplied project artifacts. New outputs produced by `src/efdt/` must be labelled as new reference-implementation runs and stored separately. The Experiment 4G script is XAI input integrity validation; attribution generation and quantitative attribution utilities are in `src/efdt/explainability.py`.

The centralized baseline is executable; the currently deposited result archive does not contain a centralized Macro-F1 or Macro-AUROC artifact. The project authors report that these experiments were performed; the original run-level evidence should be added to `results/centralized/` when available. Do not imply that the repository currently contains those metrics.

The class-wise patient adaptation figure and paired values are not currently present in this Git tree. Deposit the original source outputs before publishing exact class-wise medians/IQRs as reproducible. Analytical resource estimates must be distinguished from physical measurements.
