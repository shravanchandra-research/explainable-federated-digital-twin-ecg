# An Explainable Federated Digital Twin Framework for Adaptive and Resource-Efficient Multi-Label ECG Classification

**Repository status: partial historical research-code deposit, with a separately supplied reference implementation companion.**

This repository contains the available historical PTB-XL inspection, label analysis, preprocessing, split-validation, and HDF5 preparation scripts through Experiment 5A. A separate reference implementation archive has also been prepared from the written manuscript method. That archive is a new reconstruction; it is not recovered historical source code and it has not generated the already-supplied federated metric JSON files.

See:
- [Experiment map](docs/EXPERIMENT_MAP.md)
- [Execution guide](docs/EXECUTION.md)
- [Reproducibility statement](docs/REPRODUCIBILITY.md)
- [Reference implementation provenance and usage](docs/REFERENCE_IMPLEMENTATION.md)

## Dataset

The work uses **PTB-XL v1.0.3**, a public clinical 12-lead ECG dataset containing 21,799 ECG records from 18,869 patients. Recordings are 10 seconds long; the 100 Hz configuration has 1,000 samples per lead.

Download from [PhysioNet PTB-XL v1.0.3](https://physionet.org/content/ptb-xl/1.0.3/). Dataset DOI: [10.13026/kfzx-aw45](https://doi.org/10.13026/kfzx-aw45).

The five diagnostic superclasses are NORM, MI, STTC, CD, and HYP. The reported patient-preserving split uses folds 1–8 for training, fold 9 for validation, and fold 10 for testing.

**Raw PTB-XL data are not included.** Set the `PTBXL_ROOT` environment variable to the local extracted PTB-XL v1.0.3 directory. Do not commit raw waveform files or generated patient-level datasets.

## Reported federated metrics

The table below summarizes values in `results/federated/`. These are retained source artifacts, not independently reproduced by the current repository deposit or the separately supplied reference implementation.

| Client allocation | Optimizer | Global Macro-F1 | Global Macro-AUROC | Client F1 SD |
|---|---|---:|---:|---:|
| IID | FedAvg | 0.844234 | 0.964234 | 0.046933 |
| IID | FedAdam | 0.893406 | 0.990000 | 0.045312 |
| Non-IID | FedAvg | 0.738362 | 0.858362 | 0.021576 |
| Non-IID | FedAdam | 0.788362 | 0.908362 | 0.021576 |

A centralized reference score is not reported because a validated value was not available for this deposit.

## What the separate reference implementation adds

The companion archive supplies a new implementation of the manuscript-described CNN, centralized training, simulated-client FedAvg/FedAdam, patient-level support/query adaptation, Integrated Gradients with basic explanation audits, analytical resource accounting, and retrospective patient-state replay.

Seven unit tests pass using synthetic fixtures. They are not a full PTB-XL training run and do not reproduce the historical federated result files. The companion is documented separately in [docs/REFERENCE_IMPLEMENTATION.md](docs/REFERENCE_IMPLEMENTATION.md); its provenance must remain distinct from the original scripts and supplied metrics.

## Important scientific limits

- PTB-XL is clinical 12-lead data; this work does not establish generalization to single-lead wearable ECG.
- The Digital Twin component is retrospective patient-state replay/adaptation, not a live synchronized physiological twin.
- Federated clients in the reported simulation are simulated partitions, not independently operating hospitals.
- Federated learning alone does not establish formal privacy. Secure aggregation or differential privacy would need separate implementation and evaluation.
- Latency and energy values derived from assumed throughput and power are scenario estimates, not physical hardware measurements.
- Experiment 4G validates XAI input files; it is not an explanation-generation algorithm.
- The four federated JSON files are retained as supplied results and have not been independently recomputed from deposited training code.

## Installation of historical scripts

```bash
python -m venv .venv
# Activate the environment, then:
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

WFDB 4.3.1 is the version recorded in the workflow. Exact historical versions for the other dependencies were not captured; `requirements.txt` is not a fully locked historical environment.

For the new companion reference implementation, use the archive-specific `requirements-reference.txt` and instructions in [docs/REFERENCE_IMPLEMENTATION.md](docs/REFERENCE_IMPLEMENTATION.md).

## Citation

Cite PTB-XL as described by PhysioNet and cite the associated manuscript when its final publication details are available. No manuscript DOI or archived software DOI is assigned in this repository yet.

## License

No open-source license has been selected or granted. See [LICENSE_STATUS.md](LICENSE_STATUS.md). Public visibility does not itself grant reuse or redistribution rights.

## Maintainer

Shravanchandra G — [@shravanchandra-research](https://github.com/shravanchandra-research)
