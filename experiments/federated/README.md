# Federated experiment runners

Reserved location for manuscript-specific FedAvg/FedAdam rerun orchestration. The reusable implementation is in `src/efdt/federated.py`; supplied historical metrics are in `results/federated/`.

Preserve separate IID and non-IID configurations, patient-disjoint client assignments, client count, local epochs, communication rounds, participation rate, aggregation details, seed and per-round logs. New runs must not overwrite historical result JSON/CSV files.
