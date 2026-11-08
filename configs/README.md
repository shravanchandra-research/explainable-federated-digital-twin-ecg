# Configuration

The deposited scripts read the PTB-XL root from the `PTBXL_ROOT` environment variable. If it is not set, the original Windows path `C:\PTB-XL` is retained as a fallback for compatibility with the original experiment environment.

Windows PowerShell:

```powershell
$env:PTBXL_ROOT = "C:\PTB-XL"
```

Linux/macOS:

```bash
export PTBXL_ROOT="/path/to/PTB-XL"
```

The centralized model, federated training, adaptation, explanation-generation, and resource-profiling configuration files are not available in this deposit.
