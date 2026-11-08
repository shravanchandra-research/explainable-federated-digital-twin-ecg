# Reference implementation companion

## Status and provenance

A separate reference implementation archive accompanies this repository. It was written after the original experiment scripts and result archive were inspected, using the method and architecture described in the manuscript. It is a **new reconstruction**, not recovered historical source code and not the implementation that generated the four supplied federated metric JSON files.

The archive is not yet committed into this GitHub repository. It is supplied as a separate ZIP companion so that its provenance is not confused with the earlier experiment scripts.

## What it implements

- The manuscript's compact 1D CNN: 12-channel input, 1,000 samples, three convolutional blocks, global average pooling, and five sigmoid outputs. The implemented architecture has 14,581 trainable parameters.
- Centralized training using AdamW, binary cross-entropy with logits, patient-preserving folds 1–8 / 9 / 10, early stopping, and validation-derived thresholds.
- Simulated patient-disjoint clients with sample-weighted FedAvg and FedAdam aggregation. The Dirichlet non-IID concentration is explicitly configurable; the reference default is alpha=0.3 and must not be confused with a documented historical setting unless independently verified.
- Patient-level support/query adaptation for patients with repeated ECG recordings. Support and query samples are disjoint; the held-out query portion is used for evaluation.
- Integrated Gradients with a zero baseline and numerical integration, plus basic perturbation and attribution-stability checks.
- Analytical parameter, MAC, memory, latency, and energy accounting. Optional host-side inference timing is not a hardware energy measurement.
- Retrospective patient-state replay for the Digital Twin concept. It is not a live, continuously synchronized physiological twin.

## Data contract

The loader consumes the Experiment 5A HDF5 file and expects signal, target, patient, and fold arrays. Signals stored as (records, 1,000, 12) are transposed to (records, 12, 1,000) for the network. The implementation restricts evaluation to official PTB-XL folds: train 1–8, validation 9, test 10. Threshold selection uses validation data only.

Raw PTB-XL files and derived patient-level data are not included. Follow the PTB-XL license and PhysioNet terms; do not upload raw or patient-level records to this repository.

## Installation and commands

The archive contains its own `pyproject.toml` and `requirements-reference.txt`. From the extracted archive root:

```bash
python -m pip install -r requirements-reference.txt
python -m pip install -e .
python -m unittest discover -s tests -v
python -m efdt --help
```

Examples:

```bash
python -m efdt centralized --h5 /path/to/ptbxl_100.h5 --seed 17
python -m efdt federated --h5 /path/to/ptbxl_100.h5 --optimizer fedavg --allocation iid --seed 17
python -m efdt federated --h5 /path/to/ptbxl_100.h5 --optimizer fedadam --allocation non-iid --seed 17
python -m efdt adapt --h5 /path/to/ptbxl_100.h5 --checkpoint /path/to/checkpoint.pt
python -m efdt explain --h5 /path/to/ptbxl_100.h5 --checkpoint /path/to/checkpoint.pt --row 100 --target-class 1
python -m efdt resources --measure --device cpu
```

Use `python -m efdt <subcommand> --help` to inspect required arguments and options before running a full experiment.

## Tests and validation boundary

Seven unit tests pass in the supplied development environment. They cover model shape/parameter count, HDF5 interface and fold routing, patient-preserving simulated-client partitioning, metrics and threshold selection, Integrated Gradients and perturbation checks, analytical resource arithmetic, and retrospective patient-state replay.

These tests use synthetic fixtures and do not establish that training has been run on the complete PTB-XL dataset. Before citing results from this implementation, run the full protocol on the actual dataset, retain configuration/seed/checkpoint/prediction artifacts, and compare outputs independently. Do not substitute the reference implementation's future outputs for the supplied historical JSON metrics.

## Known limitations

- Simulated clients are not hospitals or independent institutions.
- Federated learning by itself does not provide formal privacy; differential privacy and secure aggregation are not implemented as guarantees here.
- PTB-XL is a clinical 12-lead dataset and does not prove wearable single-lead performance.
- Patient adaptation requires repeated recordings and must use genuinely disjoint support/query records.
- The Digital Twin implementation is retrospective state replay, not a validated live physiological twin.
- Resource figures computed from assumed throughput/power are analytical scenarios, not physical hardware measurements.
- The supplied reference implementation is research code, not a clinically validated diagnostic system.
