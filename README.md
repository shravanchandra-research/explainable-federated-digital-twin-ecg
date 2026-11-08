# An Explainable Federated Digital Twin Framework for Adaptive and Resource-Efficient Multi-Label ECG Classification

**Complete research package:** PTB-XL preparation scripts, author-validated result artifacts, reproducibility configuration, and the executable CNN/federated/adaptation/explainability reference implementation.

The historical result files are preserved as supplied. The reference implementation is maintained as a runnable software package and its outputs are clearly labelled separately; it does not overwrite or silently regenerate historical metrics.

## Contents

- **17 PTB-XL experiment scripts** for waveform/metadata validation, diagnostic-code and superclass analysis, patient/fold integrity, preprocessing, model-ready splits and HDF5 preparation.
- **Executable model and method modules** under `src/efdt/`: compact 1D CNN, centralized training, FedAvg/FedAdam client simulation, patient support/query adaptation, retrospective Digital Twin state replay, Integrated Gradients, explanation audits and resource accounting.
- **Author-validated result archive:** four federated JSON result files, a consolidated CSV and supplied federated-comparison/patient-adaptation figures. See [Results validation](docs/RESULTS_VALIDATION.md).
- **Eight unit/syntax tests** covering the historical scripts and reference implementation.
- Documentation, protocol configuration, workflow checks and citation metadata.

## Dataset and task

The study uses PTB-XL v1.0.3: 21,799 clinical 12-lead ECG records from 18,869 patients. Each 100 Hz, 10-second record contains 1,000 samples per lead. Five non-mutually-exclusive diagnostic superclasses are used: NORM, MI, STTC, CD and HYP.

The patient-preserving split uses folds 1–8 for training, fold 9 for validation and fold 10 for testing. Raw PTB-XL data are not included. Download from [PhysioNet PTB-XL v1.0.3](https://physionet.org/content/ptb-xl/1.0.3/) and follow its terms and citation requirements.

## Author-validated federated results

| Allocation | Optimizer | Global Macro-F1 | Global Macro-AUROC | Client F1 population SD |
|---|---|---:|---:|---:|
| IID | FedAvg | 0.844234 | 0.964234 | 0.046933 |
| IID | FedAdam | 0.893406 | 0.990000 | 0.045312 |
| Non-IID | FedAvg | 0.738362 | 0.858362 | 0.021576 |
| Non-IID | FedAdam | 0.788362 | 0.908362 | 0.021576 |

The patient adaptation archive reports a mean MI F1 change of −0.076190. The CNN resource model has 14,581 trainable parameters and 1,309,760 MACs per ECG. Its 13.10 ms latency and 6.55 mJ energy values are calculations under 100 MMAC/s and 0.5 W assumptions, not physical device measurements.

## Install and run

For the dataset preparation scripts:

```bash
python -m pip install -r requirements.txt
```

For model training, federated workflows, adaptation and explainability:

```bash
python -m pip install -r requirements-reference.txt
python -m pip install -e .
python -m unittest discover -s tests -v
python -m efdt --help
```

Set `PTBXL_ROOT` for the original scripts. The model workflows consume the Experiment 5A HDF5 output. See [Execution guide](docs/EXECUTION.md), [Experiment map](docs/EXPERIMENT_MAP.md), [Reference implementation](docs/REFERENCE_IMPLEMENTATION.md), and [Reproducibility](docs/REPRODUCIBILITY.md).

## Scientific scope

Federated clients are simulated patient-disjoint partitions, not hospitals. The Digital Twin component is retrospective patient-state replay, not a live synchronized physiological twin. PTB-XL is clinical 12-lead data and does not establish wearable single-lead performance. Federated learning alone does not establish formal privacy. Explanation outputs do not establish causal or clinical validity. Scenario-based latency and energy are not physical hardware measurements.

## Citation and license

Cite PTB-XL and the associated EFDT manuscript when its final publication details are available. No manuscript DOI or archived software DOI is asserted here. No open-source license has been granted; see [LICENSE_STATUS.md](LICENSE_STATUS.md).

Repository: https://github.com/shravanchandra-research/explainable-federated-digital-twin-ecg
