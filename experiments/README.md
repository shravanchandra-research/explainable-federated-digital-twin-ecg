# Manuscript experiment runners

This folder is reserved for manuscript-specific orchestration scripts and reviewer-requested reruns. Reusable implementations already live in `src/efdt/` and the historical PTB-XL preparation scripts remain under `scripts/`.

Suggested subfolders:
- `centralized/`: centralized baseline runner and evaluation export.
- `federated/`: controlled FedAvg/FedAdam reruns and round-level exports.
- `adaptation/`: patient adaptation runs and paired patient-level export.
- `explainability/`: cohort/selected-record attribution and audit runs.
- `resource/`: complexity and resource reports.

Do not add placeholder code that returns manuscript metrics. Each runner must call the actual implementation, save a configuration and produce its own outputs. Generated local files are ignored by Git; commit only sanitized, reproducible evidence artifacts.
