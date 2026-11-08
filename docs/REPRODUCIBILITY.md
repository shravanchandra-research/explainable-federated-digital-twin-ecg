# Reproducibility statement

## What is included

This deposit includes the available Python scripts for PTB-XL inspection, diagnostic label analysis, patient/fold validation, preprocessing, model-ready dataset preparation, and HDF5 conversion through Experiment 5A. Supplied validation logs for Experiments 1A–3C and four federated metric JSON files are also included.

## What is not included

The currently available files do not include the complete centralized CNN implementation, federated training implementation (FedAvg/FedAdam), patient adaptation/personalization implementation, full explainability method implementation, or complete resource-profiling code. There are no corresponding prediction-level outputs or comprehensive configuration files to independently recompute all manuscript metrics.

Accordingly, this repository does not claim that the manuscript's complete results are reproducible from the current deposit.

## Environment

The imported third-party libraries in the deposited scripts are listed in `requirements.txt`. WFDB 4.3.1 was recorded during the workflow. Exact versions for the other libraries and the exact Python version used for every experiment were not captured. A future reproducibility release should include a tested environment lockfile generated from a clean environment.

## Data and split

The work uses PTB-XL v1.0.3. The reported split is patient-preserving: folds 1–8 for training, fold 9 for validation, and fold 10 for testing. Do not mix or randomly resplit records without preserving patient-level grouping.

## Result provenance

Federated results are stored as the four original summary JSON artifacts supplied in the consolidated research archive. They are retained as supplied rather than regenerated. Since the corresponding training source and prediction-level data are not deposited, these metrics have not been independently recomputed in this repository.

Before treating this deposit as a full reproduction package, add the missing implementations, exact configs/random seeds, prediction-level evaluation outputs, and a tested software environment. Clearly distinguish physical measurements from scenario-derived latency/energy estimates.
