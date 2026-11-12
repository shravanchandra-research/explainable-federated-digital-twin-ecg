# Manuscript-to-repository reproduction map

This map links the manuscript's methods and reported results to the deposited implementation and the evidence that should accompany each claim. A path marked **reserved** is a planned location, not a claim that its contents already exist.

| Manuscript component | Current code/documentation | Current evidence | Revision deposit required |
|---|---|---|---|
| PTB-XL dataset description | `scripts/01_dataset_validation/`, `scripts/02_diagnostic_analysis/`, `results/validation_logs/` | Aggregate validation logs for 1A–3C | Preserve full reproducible dataset summary and environment details |
| Preprocessing and label construction | `scripts/03_initial_preprocessing/`, `scripts/04_model_ready_pipeline/` | Validation logs and source scripts | Record exact preprocessing parameters and generated-data checksums |
| Patient-level split | `scripts/02_diagnostic_analysis/experiment_2c_patient_fold_analysis.py`, `scripts/04_model_ready_pipeline/experiment_4e_model_ready_split.py`, `experiment_4f_split_integrity.py` | Patient/fold validation logs | Include machine-readable split manifest and leakage check output |
| HDF5 model-ready data | `scripts/05_hdf5_preparation/experiment_5a_prepare_hdf5.py`, `src/efdt/data.py` | HDF5 schema described in execution docs | Include a small synthetic fixture; do not commit the raw clinical dataset |
| Compact 1D CNN | `src/efdt/model.py` | Reference model tests and resource module | Preserve exact architecture configuration and model summary |
| Centralized baseline | `src/efdt/training.py`, CLI in `src/efdt/cli.py` | Executable reference workflow; historical centralized metrics are not yet in the result archive | Deposit original run configuration, checkpoint provenance, test predictions and metrics JSON supporting any manuscript values |
| Federated client allocation | `src/efdt/federated.py`, `configs/reference_protocol.json` | Reference implementation and supplied client metric vectors | Record client-to-patient assignment manifest (without direct identifiers) and partition settings |
| FedAvg / FedAdam IID and non-IID | `src/efdt/federated.py`, `results/federated/` | Four supplied JSON artifacts and CSV summaries | Add original training logs/configuration/checkpoint or prediction provenance; preserve historical JSON files unchanged |
| Client variability | `results/federated/*/metrics_*.json` | Ten client F1 values per JSON; population SD | Keep computation script and state explicitly that SD is across clients, not seeds |
| Patient-specific adaptation | `src/efdt/adaptation.py` | Supplied adaptation figure and documented mean MI ΔF1 | Deposit paired patient-level CSV, eligible-patient count, support/query definition, and summary-generation script |
| Retrospective Digital Twin replay | `src/efdt/twin.py` | Reference state-replay implementation and tests | Add a synthetic example input/output and state-transition description; do not call it a live synchronized twin |
| Explainability | `src/efdt/explainability.py` | Integrated Gradients and perturbation/stability utilities; selected-record CLI | Deposit attribution outputs and quantitative audit results only if actually run; state cohort size and protocol |
| Complexity / resource accounting | `src/efdt/resource.py` | Analytical parameter/MAC/memory calculation | Deposit JSON/CSV and calculation assumptions; separate scenario estimates from measured host timing |
| Statistical analysis | `src/efdt/metrics.py` and tests | Metric utilities | Add statistical test/CI scripts only for analyses actually reported |
| Figures and tables | `figures/`, manuscript figure references | Figure directory currently has documentation; result figures described in archive docs | Add source plotting scripts, data tables and generated figure files |
| Limitations / privacy | `docs/ARCHITECTURE.md`, `docs/REPRODUCIBILITY.md` | Scope statements | Maintain explicit limitations: simulated clients, no DP/secure aggregation, retrospective twin, no clinical validation |

## Numerical claims requiring traceable artifacts

The manuscript has reported centralized Macro-F1 0.901240 and Macro-AUROC 0.991450. The current deposited historical result archive does not contain the corresponding centralized evaluation artifact. The project authors have stated that the experiments were performed; this repository therefore reserves a place to deposit the original evidence. Until that evidence is added, the repository itself does not independently substantiate those two values.

The patient adaptation archive currently documents mean MI ΔF1 = −0.076190 and a supplied class-wise plot. Do not publish exact class-wise medians, IQRs, patient counts or confidence intervals unless they can be regenerated from the underlying paired values.

Scenario resource values of 13.10 ms and 6.55 mJ are based on 100 MMAC/s and 0.5 W assumptions. They must not be described as measured edge-device performance.

## Evidence file naming convention

Use one folder per experiment and include:
- `config.yaml` or `config.json`
- `metrics.json` and/or `metrics.csv`
- `predictions.csv` where relevant, with pseudonymous row IDs only
- `run.log`
- `manifest.json` with commit, dataset version, split, seed and software environment
- checkpoint path/hash where applicable

Do not place direct patient identifiers or raw ECG signals in public outputs.
