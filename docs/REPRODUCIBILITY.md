# Reproducibility statement

## Research package

The repository includes 17 PTB-XL dataset validation and preparation scripts, the project-validated federated and patient-adaptation result artifacts, the compact CNN and executable centralized/federated/adaptation/explainability/resource workflows, protocol configuration, tests and execution documentation.

The supplied JSON result files and figures are preserved as the project's validated artifacts. The reference implementation is maintained as an executable workflow with separately labelled outputs. It does not overwrite historical results or claim to regenerate their original metrics.

## Data and split

The study uses PTB-XL v1.0.3. The patient-preserving split is folds 1–8 for training, fold 9 for validation and fold 10 for testing. The reference workflows select class thresholds on validation data only. The raw dataset is not redistributed; users must obtain it from PhysioNet and configure the local path.

## Result provenance

Four federated result JSONs contain ten client-level F1 values and global Macro-F1/Macro-AUROC summaries for FedAvg/FedAdam under IID/non-IID allocation. The consolidated results CSV and supplied comparison plot are retained. The exact-precision CSV recomputes client F1 population standard deviations from the ten JSON values. The supplied patient adaptation plot is retained, with the reported MI mean F1 change of −0.076190. See [RESULTS_VALIDATION.md](RESULTS_VALIDATION.md).

The supplied result archive does not report a centralized reference Macro-F1 or Macro-AUROC; no value is inferred. It is also not appropriate to interpret client-level standard deviations as variability across independent random seeds.

## Environment and execution

The original dataset-preparation scripts use dependencies in `requirements.txt`; WFDB 4.3.1 is pinned from the recorded workflow. The model/reference package has a separate `requirements-reference.txt` and `pyproject.toml`. Use a clean virtual environment and record resolved versions, Python/PyTorch versions, OS, accelerator, random seeds, model checkpoints, client assignments and prediction outputs for new runs.

Run `python -m unittest discover -s tests -v` after installing both dependency sets and the package in editable mode. The tests use synthetic fixtures and check code contracts; they complement the project's validated experiment outputs.

## Scientific boundaries

Federated clients are simulated partitions, not hospitals. Federated learning alone does not confer differential privacy or secure aggregation. The Digital Twin is retrospective state replay, not a live physiological model. PTB-XL is clinical 12-lead ECG; wearable single-lead generalization is not established. Analytical latency/energy values derived from assumed throughput/power are scenarios, not physical measurements. Attribution maps do not prove clinical validity or causal reasoning.
