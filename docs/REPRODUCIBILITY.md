# Reproducibility statement

## Historical experiment deposit

This repository includes available Python scripts for PTB-XL inspection, diagnostic label analysis, patient/fold validation, preprocessing, model-ready dataset preparation, and HDF5 conversion through Experiment 5A. Supplied validation logs for Experiments 1A–3C and four federated metric JSON files are also included.

The historical source package supplied for this repository did not include the original centralized CNN training implementation, the original FedAvg/FedAdam training implementation, the patient adaptation/personalization implementation, or the complete explanation-generation and resource-profiling code. The four federated JSON files are retained as supplied artifacts, not as proof that the corresponding training code is present.

## Separately supplied reference implementation

A companion reference implementation has been written from the method and architecture described in the manuscript. It covers the compact CNN, centralized training, simulated-client FedAvg/FedAdam, patient-level support/query adaptation, Integrated Gradients and basic explanation audits, analytical resource accounting, and retrospective patient-state replay.

This companion is a **new reconstruction**, not the recovered code that generated the historical metrics. It is documented in [REFERENCE_IMPLEMENTATION.md](REFERENCE_IMPLEMENTATION.md) and supplied as a separate archive. Seven unit tests pass on synthetic fixtures. The complete PTB-XL training and evaluation pipeline has not been run from this companion in the current environment; therefore it does not independently reproduce the manuscript's reported results.

## Environment

The historical scripts' imported third-party libraries are listed in the root `requirements.txt`. WFDB 4.3.1 was recorded during the workflow. Exact versions for the other libraries and the exact Python version used for every historical experiment were not captured. The requirements file is not a fully locked historical environment.

The separate reference implementation has its own `requirements-reference.txt` and `pyproject.toml`. For a formal reproduction, create a clean environment, record the resolved package lock, Python/PyTorch versions, OS, accelerator, and deterministic settings.

## Data and split

The work uses PTB-XL v1.0.3. The reported split is patient-preserving: folds 1–8 for training, fold 9 for validation, and fold 10 for testing. Do not mix or randomly resplit records without preserving patient-level grouping.

The reference implementation chooses thresholds on validation data only and reserves the test fold for final evaluation. Patient adaptation requires genuinely disjoint support/query records for each eligible patient.

## Result provenance

Federated results are stored as four supplied summary JSON artifacts. They are retained as supplied rather than regenerated. Since the corresponding original training source and prediction-level data are not in the historical deposit, these metrics have not been independently recomputed from the original implementation.

Do not overwrite or relabel those historical results with outputs from the new reference implementation. Any new run must be recorded separately with configuration files, random seeds, checkpoints, client assignments, prediction-level outputs, and evaluation reports.

## Remaining requirements for a full empirical reproduction

1. Execute the complete reference protocol on the full PTB-XL v1.0.3 100 Hz dataset.
2. Retain all config files, seeds, package versions, client partitions, checkpoints, and prediction-level outputs.
3. Rerun centralized, FedAvg, and FedAdam experiments and compare against the supplied historical summary values without assuming they will match.
4. Document the exact repeated-record patient eligibility and support/query allocation used for adaptation.
5. Produce explanation-level outputs and quantitative faithfulness/stability results from actual held-out ECG records.
6. Measure latency and energy on named hardware, or retain all such numbers as clearly labelled analytical scenarios.

## Interpretation limits

- The federated clients are simulated partitions, not hospitals.
- Federated learning alone does not establish formal privacy; differential privacy or secure aggregation require separate implementation and evaluation.
- PTB-XL is clinical 12-lead ECG; the results do not establish wearable single-lead performance.
- The Digital Twin implementation is retrospective patient-state replay, not a live physiological twin.
- Resource calculations based on assumed throughput and power are estimates, not physical measurements.
- The reference implementation is research software, not a clinically validated diagnostic system.
