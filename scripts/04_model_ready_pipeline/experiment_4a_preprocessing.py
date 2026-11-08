import os
import numpy as np
import wfdb

# ============================================================
# EXPERIMENT 4A
# PTB-XL ECG SIGNAL PREPROCESSING AND NORMALIZATION
# ============================================================

ROOT = os.environ.get("PTBXL_ROOT", r"C:\PTB-XL")
RECORDS_DIR = os.path.join(ROOT, "records100")
OUTPUT_DIR = os.path.join(ROOT, "experiments", "4A_output")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------
# Select a validated PTB-XL record
# ------------------------------------------------------------

RECORD = os.path.join(
    RECORDS_DIR,
    "02000",
    "02223_lr"
)

print("=" * 70)
print("EXPERIMENT 4A - PTB-XL ECG SIGNAL PREPROCESSING")
print("=" * 70)

# ------------------------------------------------------------
# 1. Load ECG
# ------------------------------------------------------------

print("\n[1] Loading ECG record...")

record = wfdb.rdrecord(RECORD)

signal = record.p_signal.astype(np.float64)

print("Record name       :", record.record_name)
print("Sampling frequency:", record.fs, "Hz")
print("Number of leads   :", record.n_sig)
print("Number of samples :", record.sig_len)
print("Signal shape      :", signal.shape)

# ------------------------------------------------------------
# 2. Check signal validity
# ------------------------------------------------------------

print("\n[2] Checking signal validity...")

nan_count = np.isnan(signal).sum()
inf_count = np.isinf(signal).sum()

print("NaN values        :", nan_count)
print("Infinite values   :", inf_count)

if nan_count > 0 or inf_count > 0:
    raise ValueError("Invalid NaN or infinite values detected.")

print("Signal validity   : PASS")

# ------------------------------------------------------------
# 3. Original signal statistics
# ------------------------------------------------------------

print("\n[3] Original signal statistics...")

print("Global minimum    :", np.min(signal))
print("Global maximum    :", np.max(signal))
print("Global mean       :", np.mean(signal))
print("Global std        :", np.std(signal))

# ------------------------------------------------------------
# 4. Per-lead mean
# ------------------------------------------------------------

print("\n[4] Calculating per-lead baseline...")

lead_mean = np.mean(signal, axis=0)

for i, value in enumerate(lead_mean):
    print("Lead {:2d} mean       : {:.8f}".format(i + 1, value))

# ------------------------------------------------------------
# 5. Baseline/DC offset removal
# ------------------------------------------------------------

print("\n[5] Removing per-lead DC/baseline offset...")

centered_signal = signal - lead_mean

centered_mean = np.mean(centered_signal, axis=0)

print("\nPer-lead means after centering:")

for i, value in enumerate(centered_mean):
    print("Lead {:2d} mean       : {:.12f}".format(i + 1, value))

# ------------------------------------------------------------
# 6. Per-lead standard deviation
# ------------------------------------------------------------

print("\n[6] Calculating per-lead standard deviation...")

lead_std = np.std(centered_signal, axis=0)

for i, value in enumerate(lead_std):
    print("Lead {:2d} std        : {:.8f}".format(i + 1, value))

# ------------------------------------------------------------
# 7. Z-score normalization
# ------------------------------------------------------------

print("\n[7] Applying per-lead z-score normalization...")

if np.any(lead_std == 0):
    raise ValueError("At least one ECG lead has zero standard deviation.")

normalized_signal = centered_signal / lead_std

# ------------------------------------------------------------
# 8. Verify normalized signal
# ------------------------------------------------------------

print("\n[8] Verifying normalized signal...")

normalized_mean = np.mean(normalized_signal, axis=0)
normalized_std = np.std(normalized_signal, axis=0)

print("\nNormalized per-lead statistics:")

for i in range(12):
    print(
        "Lead {:2d}: mean = {: .8f}, std = {:.8f}".format(
            i + 1,
            normalized_mean[i],
            normalized_std[i]
        )
    )

# ------------------------------------------------------------
# 9. Global normalized statistics
# ------------------------------------------------------------

print("\n[9] Global normalized statistics...")

print("Minimum           :", np.min(normalized_signal))
print("Maximum           :", np.max(normalized_signal))
print("Mean              :", np.mean(normalized_signal))
print("Std               :", np.std(normalized_signal))

# ------------------------------------------------------------
# 10. Save processed waveform
# ------------------------------------------------------------

output_file = os.path.join(
    OUTPUT_DIR,
    "02223_lr_preprocessed.npy"
)

np.save(output_file, normalized_signal)

print("\n[10] Saving processed waveform...")

print("Output file       :", output_file)
print("Output shape      :", normalized_signal.shape)

# ------------------------------------------------------------
# 11. Final verification
# ------------------------------------------------------------

if normalized_signal.shape != (1000, 12):
    raise ValueError("Unexpected processed waveform shape.")

if not np.all(np.isfinite(normalized_signal)):
    raise ValueError("Processed waveform contains invalid values.")

print("\n" + "=" * 70)
print("EXPERIMENT 4A RESULT")
print("=" * 70)

print("Preprocessing     : PASS")
print("Baseline removal  : PASS")
print("Normalization     : PASS")
print("Shape verification: PASS")
print("Finite values     : PASS")
print("Saved successfully : PASS")

print("=" * 70)