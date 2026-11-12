# Result validation and provenance

The project owner confirms that the four federated JSON outputs retained in this repository are validated project results. Their metric values are preserved unchanged.

## Federated outputs

Four JSON files are included under `results/federated/`: FedAvg/IID, FedAvg/non-IID, FedAdam/IID and FedAdam/non-IID. Each JSON contains ten client-level F1 values and summary metrics.

| Allocation | Optimizer | Global Macro-F1 | Global Macro-AUROC | Client F1 population SD |
|---|---|---:|---:|---:|
| IID | FedAvg | 0.8442336793 | 0.9642336793 | 0.0469326861 |
| IID | FedAdam | 0.8934062800 | 0.9900000000 | 0.0453124766 |
| Non-IID | FedAvg | 0.7383621898 | 0.8583621898 | 0.0215762648 |
| Non-IID | FedAdam | 0.7883621898 | 0.9083621898 | 0.0215762648 |

Client standard deviations are calculated across ten client-level F1 values using the population convention; they are not standard deviations across independent training seeds. The exact-precision CSV is computed from the JSON source values.

## Centralized baseline

The manuscript reports Macro-F1 0.901240 and Macro-AUROC 0.991450, and the project authors state that these experiments were performed. However, the corresponding centralized metrics, predictions, configuration and run log are not currently deposited in this Git repository. The folder `results/centralized/` is reserved for the original artifacts. The repository should not be described as independently reproducing these values until those artifacts and their provenance are added.

## Patient adaptation

The project records document a mean MI F1 change of −0.076190. The class-wise adaptation plot and raw paired patient-level values are not currently present in this Git tree. Exact per-class medians, IQRs, patient counts or confidence intervals must not be inferred from an absent plot or rounded prose. Deposit the original paired values and analysis script under `results/adaptation/` when available.

## Dataset and preprocessing validation

Aggregate logs for Experiments 1A–3C accompany the preparation scripts. They cover signal structure, dataset totals, diagnostic superclass mapping, patient/fold integrity, normalization and multi-label target construction. The reported dataset contains 21,799 ECGs and 18,869 patients, with 17,418/2,183/2,198 records in train/validation/test and no patient crossing folds. The public Experiment 1B report omits the record-specific patient/demographic row.

## Analytical resource profile

The specified 1D CNN has 14,581 trainable parameters and 1,309,760 MACs per 12-lead, 1,000-sample record. Under the declared 100 MMAC/s and 0.5 W scenario assumptions, calculated operation-normalized latency is 13.0976 ms and scenario energy is 6.5488 mJ per ECG. These are analytical scenario outputs, not physical-device measurements.

## Software tests

The package test suite includes seven reference tests plus a syntax test compiling all 17 experiment scripts. It verifies the model shape/parameter count, HDF5 contract and fold routing, patient-preserving client partitioning, multi-label metrics, Integrated Gradients, analytical resource equations and retrospective state replay. Install `requirements-reference.txt`, run `pip install -e .`, and execute `python -m unittest discover -s tests -v`.

The supplied federated JSONs are retained as provided and are not rewritten by new reference-implementation runs.
