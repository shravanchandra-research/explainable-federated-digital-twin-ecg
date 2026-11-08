# Contributing

Before proposing changes:

1. Keep raw PTB-XL records, derived patient-level data, credentials, and private information out of Git.
2. Preserve patient-level separation when changing dataset splits.
3. Record dataset version, fold split, random seeds, configuration, dependency versions, and the exact command used for each experiment.
4. Distinguish independently reproduced results from copied result artifacts and from scenario-based estimates.
5. Do not claim a module is implemented unless executable source and tests are present.
6. Add tests for data shape, label alignment, leakage checks, and metric aggregation when changing the pipeline.

Open an issue before major changes to model architecture, target labels, or the patient split.
