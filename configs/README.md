# Configuration

The deposited preprocessing scripts use the `PTBXL_ROOT` environment variable to locate the local PTB-XL v1.0.3 directory.

Set it before running any experiment, for example in Windows PowerShell:

```powershell
$env:PTBXL_ROOT = "C:\PTB-XL"
```

or Linux/macOS:

```bash
export PTBXL_ROOT="/path/to/PTB-XL"
```

The model-training, federated-learning, adaptation, and explanation-generation configurations are not included because their implementation/configuration files were not present in the available source package.
