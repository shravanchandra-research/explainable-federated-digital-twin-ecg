# Explainable Federated Digital Twin for Adaptive and Resource-Efficient Multi-Label ECG Classification

**Repository status:** Initial research repository scaffold. Source-code implementation and reproducibility materials are not yet deposited.

This repository is intended to host the code, configuration, documentation, and selected result artifacts for the research manuscript:

> *An Explainable Federated Digital Twin Framework for Adaptive and Resource-Efficient Multi-Label ECG Classification*

## Research scope

The project investigates a simulation-based framework combining:

- Multi-label ECG classification using a compact 1D convolutional neural network.
- Federated training with FedAvg and FedAdam under IID and non-IID client partitions.
- Patient-level adaptation experiments associated with a retrospective Digital Twin concept.
- Explainability analysis for model predictions.
- Analytical estimates of model memory, computation, communication, latency, and energy.

The current study uses the PTB-XL 1.0.3 12-lead ECG dataset. The dataset contains 21,799 ECG records from 18,869 patients. Records are represented as 10-second, 12-lead signals; the 100 Hz sampling configuration yields 1,000 samples per lead.

## Important scope and reproducibility notes

This repository must not be interpreted as evidence of clinical deployment or a validated real-time wearable system.

- The Digital Twin component is currently framed as retrospective patient-state replay/adaptation, not a live, continuously synchronized physiological twin.
- Federated clients in the experiments are simulated partitions, not independently operating hospitals.
- PTB-XL is a clinical 12-lead ECG dataset; results on it do not establish generalization to single-lead wearable devices.
- Federated learning by itself does not provide a formal privacy guarantee. Secure aggregation or differential privacy would require separate implementation and evaluation.
- Latency and energy values derived from assumed throughput and power are scenario estimates, not physical hardware measurements.
- Result summaries are not a substitute for source code, configuration files, random seeds, environment specifications, and executable instructions.

## Dataset

PTB-XL v1.0.3 is available from PhysioNet:

https://physionet.org/content/ptb-xl/1.0.3/

Dataset DOI: https://doi.org/10.13026/kfzx-aw45

Do not commit restricted, identifiable, or unnecessary copies of patient data. Follow the dataset's access terms and citation requirements.

## Planned repository structure

```text
.
├── README.md
├── CITATION.cff
├── LICENSE
├── requirements.txt
├── configs/
├── data/
│   └── README.md
├── src/
│   ├── data/
│   ├── models/
│   ├── federated/
│   ├── adaptation/
│   ├── explainability/
│   └── profiling/
├── experiments/
├── results/
│   └── README.md
├── figures/
├── tests/
└── docs/
```

This is a planned structure; directories and implementation files will be added as the actual research code is prepared. No placeholder script is presented as an implemented experiment.

## Experimental design (reported in the manuscript)

The reported federated configurations compare FedAvg and FedAdam under IID and non-IID data allocation. The five multi-label diagnostic superclasses are:

- NORM — Normal ECG
- MI — Myocardial infarction
- STTC — ST/T change
- CD — Conduction disturbance
- HYP — Hypertrophy

The patient-preserving split uses folds 1–8 for training, fold 9 for validation, and fold 10 for testing.

## Reported federated summary

The following values are transcribed from the experiment JSON summaries used during manuscript preparation. They have not yet been independently reproduced from code deposited in this repository.

| Client allocation | Optimizer | Global Macro-F1 | Global Macro-AUROC | Population SD of client F1 |
|---|---|---:|---:|---:|
| IID | FedAvg | 0.844234 | 0.964234 | 0.046933 |
| IID | FedAdam | 0.893406 | 0.990000 | 0.045312 |
| Non-IID | FedAvg | 0.738362 | 0.858362 | 0.021576 |
| Non-IID | FedAdam | 0.788362 | 0.908362 | 0.021576 |

A centralized reference is mentioned in the manuscript, but a validated centralized score has not been supplied for this README and is therefore not reported here.

## Current limitations of this deposit

At the time of this initial commit, the repository does **not** contain the actual Python source files or notebooks required to reproduce the experiments. The earlier consolidated archive reviewed during preparation contained experiment result JSON/CSV files and plots, but no `.py` files or `.ipynb` notebooks. Consequently:

- No end-to-end reproduction claim is made.
- No installation or execution command is given until dependencies and entry points are confirmed from the actual code.
- Explainability metrics/plots, centralized baseline results, and detailed adaptation statistics should only be added when their underlying validated artifacts are available.

## Citation

Please cite the associated manuscript when it is published. The formal citation and DOI will be added here after publication. The repository's versioned archive DOI will be added after a release is archived.

## Contact

Repository maintainer: **Shravanchandra G**  
GitHub: https://github.com/shravanchandra-research

## License

No license has been selected yet. Until a license file is added, all rights are reserved by the respective copyright holder; do not assume that the contents may be reused or redistributed.
