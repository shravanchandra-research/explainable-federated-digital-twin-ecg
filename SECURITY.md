# Security and data handling

Do not commit passwords, access tokens, private keys, patient-identifiable information, raw ECG waveforms, or private institutional records.

If a secret was committed accidentally, removing it in a later commit is not sufficient; revoke/rotate it and follow GitHub's guidance for sensitive-data removal.

For data issues, first determine whether the file contains raw patient data, metadata that could be identifying, credentials, or generated outputs before opening a pull request.
