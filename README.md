# An Explainable Federated Digital Twin Framework for Adaptive and Resource-Efficient Multi-Label ECG Classification

**Repository status: partial research-code deposit.** The PTB-XL validation, label-analysis, preprocessing, split-validation, and HDF5 preparation scripts through Experiment 5A are included. Centralized/federated model training, patient-adaptation source, and the full explainability implementation are not yet deposited.

This repository accompanies the research work titled:

> *An Explainable Federated Digital Twin Framework for Adaptive and Resource-Efficient Multi-Label ECG Classification*

## What is included

- Python scripts for Experiments 1A–3C: waveform and metadata validation, diagnostic-code analysis, superclass mapping, patient/fold integrity, initial signal preprocessing, target construction, and dataset-loader checks.
- Python scripts for Experiments 4A–4H: signal and batch preprocessing, preprocessed-dataset preparation, label alignment, model-ready split generation, split integrity, XAI-input validation, and full preprocessing.
- Python script for Experiment 5A: HDF5 preparation.
- Four supplied federated metric JSON artifacts.
- Supplied validation reports for Experiments 1A–3C.
- Research documentation, dependency list, data-handling instructions, and citation metadata.

See [docs/EXPERIMENT_MAP.md](docs/EXPERIMENT_MAP.md) for the exact file inventory and [docs/EXECUTION.md](docs/EXECUTION.md) for setup and execution instructions.

## Dataset

The work uses **PTB-XL v1.0.3**, a public clinical 12-lead ECG dataset containing 21,799 ECG records from 18,869 patients. Recordings are 10 seconds long; the 100 Hz configuration has 1,000 samples per lead.

Download from [PhysioNet PTB-XL v1.0.3](https://physionet.org/content/ptb-xl/1.0.3/). Dataset DOI: [10.13026/kfzx-aw45](https://doi.org/10.13026/kfzx-aw45).

The five diagnostic superclasses are NORM, MI, STTC, CD, and HYP. The reported patient-preserving split uses folds 1–8 for training, fold 9 for validation, and fold 10 for testing.

**Raw PTB-XL data are not included.** Set the `PTBXL_ROOT` environment variable to the local extracted PTB-XL v1.0.3 directory. Do not commit raw waveform files or generated patient-level datasets.

## Reported federated metrics

The table below summarizes values in `results/federated/`. These are retained source artifacts, not independently reproduced by the current repository deposit.

| Client allocation | Optimizer | Global Macro-F1 | Global Macro-AUROC | Client F1 SD |
|---|---|---:|---:|---:|
| IID | FedAvg | 0.844234 | 0.964234 | 0.046933 |
| IID | FedAdam | 0.893406 | 0.990000 | 0.045312 |
| Non-IID | FedAvg | 0.738362 | 0.858362 | 0.021576 |
| Non-IID | FedAdam | 0.788362 | 0.908362 | 0.021576 |

A centralized reference score is not reported because a validated value was not available for this deposit.

## Important limits

- PTB-XL is clinical 12-lead data; this work does not establish generalization to single-lead wearable ECG.
- The Digital Twin component is retrospective patient-state replay/adaptation, not a live synchronized physiological twin.
- Federated clients in the reported simulation are simulated partitions, not independently operating hospitals.
- Federated learning alone does not establish formal privacy. Secure aggregation or differential privacy would need separate implementation and evaluation.
- Latency and energy values derived from assumed throughput and power are scenario estimates, not physical hardware measurements.
- Experiment 4G validates XAI input files; it is not an explanation-generation algorithm.
- The repository does not yet contain centralized CNN training, FedAvg/FedAdam training, patient-level adaptation, or Integrated Gradients/explanation-generation source. The current deposit therefore does not reproduce all manuscript results end-to-end.

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the current boundary of reproducibility.

## Installation

```bash
python -m venv .venv
# Activate the environment, then:
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

WFDB 4.3.1 is the version recorded in the workflow. Exact historical versions for the other dependencies were not captured; `requirements.txt` is not a fully locked historical environment.

## Citation

Cite PTB-XL as described by PhysioNet and cite the associated manuscript when its final publication details are available. No manuscript DOI or archived software DOI is assigned in this repository yet.

## License

No open-source license has been selected or granted. See [LICENSE_STATUS.md](LICENSE_STATUS.md). Public visibility does not itself grant reuse or redistribution rights.

## Maintainer

Shravanchandra G — [@shravanchandra-research](https://github.com/shravanchandra-research)
