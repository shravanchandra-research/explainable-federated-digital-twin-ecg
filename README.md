# An Explainable Federated Digital Twin Framework for Adaptive and Resource-Efficient Multi-Label ECG Classification

Research code and reproducibility materials for the EFDT study using PTB-XL v1.0.3.

> **Repository status:** This repository contains PTB-XL validation/preparation scripts (Experiments 1A–5A), an executable Python reference package, supplied federated result artifacts, aggregate validation logs, tests and manuscript-oriented documentation. Some manuscript results and their original run-level evidence still need to be deposited when the corresponding source outputs are available. A directory scaffold is not evidence that an experiment has been implemented or reproduced.

**Repository:** https://github.com/shravanchandra-research/explainable-federated-digital-twin-ecg

## 1. Study at a glance

The framework investigates multi-label classification of 12-lead ECG signals using a compact 1D CNN, simulated federated learning (FedAvg and FedAdam), patient-specific adaptation, retrospective patient-state replay, explainability utilities and resource accounting.

### Dataset and task

- Dataset: [PTB-XL v1.0.3](https://physionet.org/content/ptb-xl/1.0.3/)
- Dataset scale: 21,799 ECG records from 18,869 patients.
- Signal representation: 12 leads, 100 Hz, 10 seconds (1,000 samples per lead).
- Target superclasses: NORM, MI, STTC, CD and HYP (multi-label; labels are not mutually exclusive).
- Patient-preserving split: folds 1–8 training, fold 9 validation, fold 10 testing.
- Expected split counts from the validation logs: 17,418 training, 2,183 validation and 2,198 test records.

Raw PTB-XL files are not included. Obtain the dataset from PhysioNet and comply with its access, use and citation conditions.

## 2. Repository contents

| Path | Purpose | Current status |
|---|---|---|
| `scripts/01_dataset_validation/` | Experiments 1A–1B: waveform and metadata checks | Deposited scripts |
| `scripts/02_diagnostic_analysis/` | Experiments 2A–2C: diagnostic labels, superclass and patient/fold analysis | Deposited scripts |
| `scripts/03_initial_preprocessing/` | Experiments 3A–3C: preprocessing, target construction and loader checks | Deposited scripts |
| `scripts/04_model_ready_pipeline/` | Experiments 4A–4H: preprocessing, label alignment, split integrity and XAI-input validation | Deposited scripts |
| `scripts/05_hdf5_preparation/` | Experiment 5A: HDF5 preparation | Deposited script |
| `src/efdt/` | CNN, training, federated simulation, adaptation, retrospective twin, explainability, metrics, resource accounting and CLI | Executable reference package |
| `configs/` | Manuscript-aligned reference protocol | Configuration scaffold |
| `results/federated/` | Four supplied FedAvg/FedAdam IID/non-IID result JSONs and CSV summaries | Supplied historical artifacts |
| `results/validation_logs/` | Aggregate logs for Experiments 1A–3C | Supplied aggregate logs |
| `results/centralized/` | Centralized baseline result evidence | Reserved; deposit original run evidence |
| `results/adaptation/` | Patient-level adaptation outputs | Reserved; supplied plot is described in results documentation; deposit raw paired values if available |
| `results/explainability/` | Explanation outputs and quantitative audits | Reserved for run outputs |
| `results/resource/` | Complexity and resource analysis outputs | Reserved for generated reports |
| `checkpoints/` | Model checkpoints and provenance | Reserved; do not commit large/unverified weights |
| `docs/` | Architecture, execution, experiment mapping, reproducibility, validation and revision guidance | Documentation |
| `tests/` | Reference implementation and script syntax tests | Deposited tests |

See [Experiment map](docs/EXPERIMENT_MAP.md), [Execution guide](docs/EXECUTION.md), [Manuscript-to-code map](docs/MANUSCRIPT_REPRODUCTION_MAP.md) and [Revision workflow](docs/REVISION_WORKFLOW.md).

## 3. Supplied federated result artifacts

The four JSON files under `results/federated/` contain the supplied client-level F1 values and global metrics.

| Allocation | Optimizer | Global Macro-F1 | Global Macro-AUROC | Client F1 population SD |
|---|---|---:|---:|---:|
| IID | FedAvg | 0.844234 | 0.964234 | 0.046933 |
| IID | FedAdam | 0.893406 | 0.990000 | 0.045312 |
| Non-IID | FedAvg | 0.738362 | 0.858362 | 0.021576 |
| Non-IID | FedAdam | 0.788362 | 0.908362 | 0.021576 |

These are preserved project result artifacts. Client population SD describes dispersion across the ten simulated clients; it is not variation across independent training seeds. See [Results validation](docs/RESULTS_VALIDATION.md).

### Other manuscript quantities and evidence status

- Patient adaptation: project records document a mean MI F1 change of −0.076190. The raw paired patient-level CSV and class-wise adaptation plot are not currently present in this Git tree. Deposit the original values and generating script before claiming exact class-wise medians/IQRs as reproducible.
- Centralized baseline: the manuscript reports Macro-F1 0.901240 and Macro-AUROC 0.991450. The currently deposited historical result archive does not yet contain the corresponding centralized predictions/metrics artifact. Deposit the original evaluation output, test predictions, configuration and checkpoint provenance when available. Do not imply that the repository currently reproduces this value.
- Model complexity: 14,581 trainable parameters and 1,309,760 MACs per ECG are documented in the project workflow.
- Resource scenario: 13.10 ms and 6.55 mJ are analytical calculations under 100 MMAC/s and 0.5 W assumptions, not physical-device measurements.

## 4. Installation

Use a clean virtual environment. Python 3.10 or newer is specified for the reference package.

### Dataset preparation scripts

```bash
python -m pip install -r requirements.txt
```

### Model/reference package

```bash
python -m pip install -r requirements-reference.txt
python -m pip install -e .
python -m efdt --help
```

The two requirements files serve different parts of the repository. For a combined environment, install both.

## 5. Dataset setup

Download and extract PTB-XL v1.0.3 locally. Set `PTBXL_ROOT` to the extracted dataset directory.

Windows PowerShell:

```powershell
$env:PTBXL_ROOT = "C:\PTB-XL"
```

Linux/macOS:

```bash
export PTBXL_ROOT="/path/to/PTB-XL"
```

The directory should contain `ptbxl_database.csv`, `scp_statements.csv`, `RECORDS` and the relevant waveform directory (for example, `records100/`). Consult [Execution guide](docs/EXECUTION.md) for the complete preparation sequence and HDF5 contract.

## 6. Run tests

```bash
python -m unittest discover -s tests -v
```

The tests use synthetic fixtures to check software contracts; they do not independently validate clinical performance or regenerate all historical manuscript results.

## 7. Scientific scope and limitations

- Federated clients are simulated patient-disjoint partitions, not real hospitals or independent clinical institutions.
- Federated learning alone does not provide formal differential privacy or secure aggregation.
- The Digital Twin component is retrospective computational patient-state replay, not a continuously synchronized physiological twin.
- PTB-XL is clinical 12-lead ECG data; wearable single-lead generalization is not established by this dataset.
- Integrated Gradients and perturbation/stability utilities are attribution analyses; they do not establish causal or clinical validity.
- Scenario-based latency and energy calculations are not physical-device measurements.
- The repository is research software, not a medical device or a clinical decision system.

## 8. Reproducibility and result provenance

Do not overwrite supplied historical result artifacts with outputs from a later reference-code run. Store each new run separately and preserve:
1. Git commit or release identifier;
2. dataset version and split;
3. configuration and random seed;
4. Python, PyTorch and dependency versions;
5. checkpoint identifier/hash;
6. predictions and metrics;
7. client allocation and aggregation settings for federated runs.

Use `results/experiment_manifest.template.json` as a starting point. Record unavailable fields as null or explain why they are unavailable; never invent provenance.

## 9. Citation, licensing and data use

Citation metadata is provided in [CITATION.cff](CITATION.cff). Update it when the manuscript receives final publication details and when a software release is archived.

No open-source license is currently granted. See [LICENSE_STATUS.md](LICENSE_STATUS.md). Public visibility does not grant reuse rights. Do not add a license until all contributors confirm authorization.

PTB-XL remains governed by its own dataset terms. This repository does not redistribute the raw dataset.

## 10. Revision contributions

During peer review, add new implementations to the existing `src/efdt/` package and experiment-specific entry points under `experiments/` or the reserved `scripts/` stages. Add generated outputs under the corresponding `results/` subfolder. Update the manuscript-to-code map and revision log with each substantive change.

See [Revision workflow](docs/REVISION_WORKFLOW.md) for the expected process.
