# Revision and experiment-deposit workflow

This document reserves the repository workflow for peer-review revisions. It does not imply that planned experiments have already been implemented.

## 1. Before starting a revision experiment

1. Create a branch named `revision/<reviewer-or-experiment>`.
2. Record the current commit SHA and do not modify the supplied historical result files.
3. Confirm the PTB-XL version, patient-preserving folds, class order and exact manuscript claim being tested.
4. Copy the relevant protocol configuration and record any deviations.
5. Use a new output directory; never silently replace a previous run.

## 2. Where to add code

- Reusable model/data/method code: `src/efdt/`.
- Manuscript-specific experiment runners: `experiments/<stage>/`.
- Existing historical preparation scripts: retain under `scripts/01_...` through `scripts/05_...`.
- Experiment configuration: `configs/`.
- Machine-readable results: `results/<stage>/`.
- Figures/tables and their generating scripts: `figures/` and the corresponding `results/<stage>/` directory.
- Tests: `tests/`.

The repository's `.gitignore` intentionally excludes generated local outputs and checkpoints. Commit small, sanitized evidence artifacts. Use a separate approved archive for large checkpoints and record immutable hashes/URLs.

## 3. Required run manifest

For every new run, preserve:
- experiment identifier and UTC timestamp;
- Git commit SHA and branch/tag;
- dataset name/version and local preprocessing artifact identifier;
- train/validation/test fold definition;
- class order and label mapping;
- model architecture and parameter count;
- optimizer, learning rate, batch size, epochs, early stopping and threshold-selection protocol;
- random seed(s), Python/PyTorch/library versions, operating system and device;
- federated client count, patient partition method, Dirichlet alpha, local epochs, rounds, participation and aggregation settings, if applicable;
- patient adaptation eligibility and disjoint support/query construction, if applicable;
- XAI method, baseline, integration steps and faithfulness/stability protocol, if applicable;
- output file names, checkpoint hash and metrics.

Use `results/experiment_manifest.template.json` as the starting point. Use null for unavailable values and explain the omission.

## 4. Required evidence by experiment

| Experiment | Minimum evidence |
|---|---|
| Centralized baseline | Config, training log, selected checkpoint, held-out test predictions, per-class metrics, Macro-F1 and Macro-AUROC |
| FedAvg/FedAdam | Config, client partition manifest, round logs, aggregation settings, global predictions/metrics and client metrics |
| Patient adaptation | Eligible-patient count, per-patient support/query sizes, paired before/after predictions and class-wise ΔF1 |
| Digital Twin replay | State schema, update rules, synthetic replay trace and tests |
| Explainability | Selected ECG identifiers pseudonymized, attribution arrays/figures, target class, baseline/steps, faithfulness and stability outputs |
| Resource analysis | Parameter/MAC/memory calculation, assumptions, device details and measured latency/energy protocol if hardware measurements are claimed |
| Statistical comparison | Predefined comparison, sample unit, test, assumptions, effect size, confidence interval and corrected p-values where multiple tests apply |

## 5. Result integrity rules

- Never replace the supplied historical JSON/CSV artifacts with a later run.
- Never present a newly generated result as the original historical result.
- Do not infer missing metrics from plots or rounded tables.
- Do not claim statistical significance without an appropriate analysis.
- Do not call simulated clients hospitals.
- Do not call retrospective replay a live physiological twin.
- Do not call analytical latency/energy estimates hardware measurements.
- Do not claim formal privacy from federated learning alone.
- Do not upload raw PTB-XL files, direct identifiers, secrets or unnecessary patient-level data.

## 6. Reviewer response and release

1. Link each reviewer comment to a code/config/result change.
2. Run tests and document environment.
3. Regenerate only the affected tables/figures and explain any changed results.
4. Update `docs/MANUSCRIPT_REPRODUCTION_MAP.md` and the README.
5. Commit the revision with a descriptive message.
6. After acceptance/publication, tag a stable release and archive that exact commit (for example with Zenodo) if the authors choose to do so.
