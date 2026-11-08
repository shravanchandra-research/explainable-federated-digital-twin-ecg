# Executable reference implementation

The `src/efdt/` package provides executable modules for the study's compact CNN and integrated method:

- Centralized training with AdamW, BCE-with-logits, validation early stopping and validation-only threshold selection.
- Patient-preserving simulated-client partitions with IID/non-IID allocation, FedAvg and FedAdam.
- Patient-specific support/query adaptation on held-out patients.
- Retrospective computational patient-state replay.
- Integrated Gradients, perturbation-faithfulness and attribution-stability utilities.
- Analytical parameter, MAC, memory, latency and energy accounting, plus optional host-device inference timing.

## HDF5 data contract

The reader expects the Experiment 5A HDF5 to contain `X` with shape `(N,1000,12)`, `Y` with shape `(N,5)` in NORM/MI/STTC/CD/HYP order, and the metadata arrays `ecg_id`, `patient_id` and `fold`. Inputs are transposed to `(12,1000)`. Folds 1–8 are used for training, fold 9 for validation and fold 10 for final testing.

## Install and test

```bash
python -m pip install -r requirements-reference.txt
python -m pip install -e .
python -m unittest discover -s tests -v
python -m efdt --help
```

Seven reference unit tests use synthetic fixtures; the historical experiment scripts have a separate syntax-compile test. The supplied result files in `results/` are retained independently and are not overwritten by reference-code runs.

## Methodological scope

Simulated clients are not real hospitals. Adaptation uses deterministic ECG-ID ordering, not an asserted clinical timestamp sequence. The Twin stores retrospective state and is not a live physiological twin. Integrated Gradients and perturbation/stability measures are attribution diagnostics, not clinical validity. The throughput/power latency and energy calculations are analytical scenarios; optional timing is measured only on the selected host device. The code does not claim formal differential privacy, secure aggregation, wearable validation or clinical deployment.
