# Tests

Run the lightweight syntax test from the repository root:

```bash
python -m unittest discover -s tests
```

The test compiles each deposited `experiment_*.py` file without executing it or loading PTB-XL. A passing syntax test does not verify data availability, scientific correctness, metric validity, or end-to-end reproducibility.

The next testing layer should cover waveform dimensions and finite values, diagnostic label mapping, patient-level split isolation, index/file alignment, HDF5 shape/dtype/content, and metric aggregation.
