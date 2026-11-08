# An Explainable Federated Digital Twin Framework for Adaptive and Resource-Efficient Multi-Label ECG Classification

**Repository status: partial research-code deposit.** This repository now contains the available PTB-XL dataset inspection, label analysis, preprocessing, split-validation, and HDF5 preparation scripts for Experiments 1A–5A, selected validation logs, and the supplied federated metric JSON summaries. It does not yet contain all code needed to reproduce the paper's model-training, adaptation, and explainability results.

## Research overview

The project investigates a simulation-based framework combining multi-label ECG classification, federated learning, a retrospective patient-state adaptation concept, explainability, and resource analysis.

**Dataset:** PTB-XL v1.0.3, a clinical 12-lead ECG dataset (21,799 records from 18,869 patients). The 100 Hz records contain 1,000 samples per lead over 10 seconds. The five diagnostic superclasses are NORM, MI, STTC, CD, and HYP.

**Patient-preserving split:** folds 1–8 training, fold 9 validation, fold 10 test.

## Repository contents

| Directory | Contents |
|---|---|
| `scripts/01_dataset_validation/` | Experiments 1A–1B: waveform and metadata checks |
| `scripts/02_diagnostic_analysis/` | Experiments 2A–2C: diagnostic codes, superclasses, patient/fold analysis |
| `scripts/03_initial_preprocessing/` | Experiments 3A–3C: preprocessing, labels, dataset-loader validation |
| `scripts/04_model_ready_pipeline/` | Experiments 4A–4H: preprocessing, alignment, split/integrity validation, XAI input validation, full dataset preparation |
| `scripts/05_hdf5_preparation/` | Experiment 5A: HDF5 preparation |
| `results/validation_logs/` | Supplied text reports for Experiments 1A–3C |
| `results/federated/` | Four original federated metric JSON summaries |
| `docs/` | Execution, architecture, experiment inventory and reproducibility statements |
| `configs/` | Dataset-root configuration guidance |

The included scripts have been syntax-compiled after the portable data-root change. This does not mean that every experiment has been rerun from a clean environment.

## Setup and execution

1. Download **PTB-XL v1.0.3** from [PhysioNet](https://physionet.org/content/ptb-xl/1.0.3/).
2. Extract it locally; do not commit the raw dataset to this repository.
3. Install dependencies using `pip install -r requirements.txt`.
4. Set `PTBXL_ROOT` to the extracted PTB-XL directory.
5. Follow the experiment order and prerequisites in [docs/EXECUTION.md](docs/EXECUTION.md).

For Windows PowerShell:

```powershell
$env:PTBXL_ROOT = "C:\PTB-XL"
python scripts/01_dataset_validation/experiment_1a_ptbxl_verify.py
```

See [docs/EXPERIMENT_MAP.md](docs/EXPERIMENT_MAP.md) for the current coverage and [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for limitations.

## Federated result summaries supplied with the project

| Allocation | Optimizer | Global Macro-F1 | Global Macro-AUROC | Client F1 population SD |
|---|---|---:|---:|---:|
| IID | FedAvg | 0.844234 | 0.964234 | 0.046933 |
| IID | FedAdam | 0.893406 | 0.990000 | 0.045312 |
| Non-IID | FedAvg | 0.738362 | 0.858362 | 0.021576 |
| Non-IID | FedAdam | 0.788362 | 0.908362 | 0.021576 |

These values are copied from the supplied experiment JSON summaries into `results/federated/`. They are **reported artifacts, not independently reproduced by the scripts currently deposited here**. The consolidated archive did not contain the training implementation or prediction-level outputs used to recompute these metrics. No centralized score is reported because a validated value was not supplied.

## Critical scope and limitations

- This is a simulation study, not a clinical deployment or prospective validation.
- The Digital Twin component is retrospective patient-state replay/adaptation, not a live synchronized physiological twin.
- Federated clients are simulated data partitions, not independently operating hospitals.
- Federated learning alone does not establish formal privacy. Secure aggregation or differential privacy would need separate implementation and evaluation.
- PTB-XL is clinical 12-lead ECG; this dataset does not establish single-lead wearable generalization.
- Reported latency/energy estimates based on assumed throughput and power are scenario estimates, not physical hardware measurements.
- Experiment 4G validates XAI input readiness; it is not itself an explanation-generation or XAI faithfulness experiment.
- Centralized CNN training, FedAvg/FedAdam training, patient adaptation, explanation generation/evaluation, and complete resource-profiling source code are not yet included.

## Dataset citation

Wagner, P. et al. (2020). *PTB-XL, a large publicly available electrocardiography dataset*. Scientific Data. Dataset version 1.0.3: [PhysioNet](https://physionet.org/content/ptb-xl/1.0.3/), [DOI](https://doi.org/10.13026/kfzx-aw45).

## Citation and license

See [CITATION.cff](CITATION.cff). The manuscript DOI and archived software release DOI will be added only after they have actually been assigned.

**No open-source license has been granted yet.** Public visibility does not itself grant reuse rights. See [LICENSE_STATUS.md](LICENSE_STATUS.md). A license must be approved by the code contributors before publication under an open-source license.

## Maintainer

Shravanchandra G — [GitHub profile](https://github.com/shravanchandra-research)
