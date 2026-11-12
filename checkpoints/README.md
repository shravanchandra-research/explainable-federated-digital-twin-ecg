# Model checkpoint policy

This folder is reserved for model checkpoint references and small approved artifacts. Large checkpoint binaries are excluded by `.gitignore`; use an approved archival location or Git LFS only after deciding on storage and access requirements.

For every checkpoint referenced by a manuscript result, record:
- experiment/run ID;
- architecture and configuration;
- dataset version and split;
- Git commit;
- checkpoint SHA-256;
- training seed and selected epoch;
- associated evaluation metrics and prediction artifact location.

Do not upload unverified weights, raw patient data, or checkpoints whose provenance cannot be established.
