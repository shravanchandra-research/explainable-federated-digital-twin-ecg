# Result validation and provenance

The project owner confirms that the outputs retained from the consolidated EFDT result archive are validated project results. The original JSON metrics, CSV and figures are carried forward without editing the historical values.

## Federated outputs

Four JSON files are included under `results/federated/`: FedAvg/IID, FedAvg/non-IID, FedAdam/IID and FedAdam/non-IID. Each JSON contains ten client-level F1 values, global Macro-F1, global Macro-AUROC, client-to-client population standard deviation, transmitted data and average training time per round.

| Allocation | Optimizer | Global Macro-F1 | Global Macro-AUROC | Client F1 population SD |
|---|---|---:|---:|---:|
| IID | FedAvg | 0.8442336793 | 0.9642336793 | 0.0469326861 |
| IID | FedAdam | 0.8934062800 | 0.9900000000 | 0.0453124766 |
| Non-IID | FedAvg | 0.7383621898 | 0.8583621898 | 0.0215762648 |
| Non-IID | FedAdam | 0.7883621898 | 0.9083621898 | 0.0215762648 |

Client standard deviations are calculated across ten client-level F1 values using the population convention; they are not standard deviations across independent training seeds. The exact-precision CSV is computed from the JSON source values.

## Patient adaptation

The supplied patient-adaptation plot is stored in the complete research package. The documented mean MI F1 change is −0.076190. The remaining class distributions are interpreted descriptively; exact per-class summaries are not invented where raw values are not available.

## Dataset and preprocessing validation

Aggregate logs for Experiments 1A–3C accompany the preparation scripts. They cover signal structure, dataset totals, diagnostic superclass mapping, patient/fold integrity, normalization and multi-label target construction. The reported dataset contains 21,799 ECGs and 18,869 patients, with 17,418/2,183/2,198 records in train/validation/test and no patient crossing folds. The public Experiment 1B report omits the record-specific patient/demographic row.

## Analytical resource profile

The specified 1D CNN has 14,581 trainable parameters and 1,309,760 MACs per 12-lead, 1,000-sample record. Under the declared 100 MMAC/s and 0.5 W scenario assumptions, calculated operation-normalized latency is 13.0976 ms and scenario energy is 6.5488 mJ per ECG. These are analytical scenario outputs, not physical-device measurements.

## Software tests

The complete package test suite includes seven reference tests plus a syntax test compiling all 17 experiment scripts. It verifies the model shape/parameter count, HDF5 contract and fold routing, patient-preserving client partitioning, multi-label metrics, Integrated Gradients, analytical resource equations and retrospective state replay. Install `requirements-reference.txt`, run `pip install -e .`, and execute `python -m unittest discover -s tests -v`.

The original result JSONs are retained as supplied and are not rewritten by new reference-implementation runs. The consolidated result archive does not report centralized baseline Macro-F1/Macro-AUROC values; none are inferred.
