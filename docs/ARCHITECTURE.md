# Research workflow architecture

This diagram describes the intended study workflow and identifies which parts are currently represented by deposited source code. It is a conceptual pipeline, not proof that every stage is implemented in this repository.

```mermaid
flowchart TD
  A[PTB-XL v1.0.3: 12-lead ECG + metadata] --> B[Waveform and metadata validation]
  B --> C[Diagnostic SCP code analysis]
  C --> D[Five superclass targets: NORM / MI / STTC / CD / HYP]
  D --> E[Patient-preserving folds 1-8 / 9 / 10]
  E --> F[Preprocessing and record alignment]
  F --> G[Model-ready dataset / HDF5]
  G --> H[Centralized CNN training]
  G --> I[Federated training: FedAvg / FedAdam]
  H --> J[Evaluation and patient-level analysis]
  I --> J
  J --> K[Patient adaptation / retrospective Digital Twin]
  J --> L[Explainability and resource analysis]
```

## Deposited implementation coverage

The source code currently deposited covers waveform/metadata inspection, label mapping, patient/fold analysis, preprocessing, alignment, split validation, XAI input-file validation, full preprocessed dataset creation, and HDF5 preparation (Experiments 1A–5A).

The available files do not include centralized or federated model-training implementations, patient-level adaptation code, explanation generation (e.g. Integrated Gradients), or complete resource-profiling source. Those blocks are therefore shown as intended downstream workflow components, not as implemented repository modules.
