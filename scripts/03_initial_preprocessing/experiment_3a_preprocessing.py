import os
import wfdb
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# EXPERIMENT 3A
# PTB-XL ECG PREPROCESSING PIPELINE
# ============================================================

BASE_DIR = Path(os.environ.get("PTBXL_ROOT", r"C:\PTB-XL"))

RECORD_PATH = BASE_DIR / "records100" / "01000" / "01000_lr"

OUTPUT_DIR = BASE_DIR / "experiments"

REPORT_FILE = OUTPUT_DIR / "experiment_3a_preprocessing.txt"

PLOT_FILE = OUTPUT_DIR / "experiment_3a_preprocessing.png"

# ============================================================
# 1. LOAD ECG
# ============================================================

print("=" * 70)
print("EXPERIMENT 3A - PTB-XL ECG PREPROCESSING PIPELINE")
print("=" * 70)

print()
print("Loading ECG record...")

record = wfdb.rdrecord(str(RECORD_PATH))

signal = record.p_signal.astype(np.float32)

print("Record:", record.record_name)
print("Sampling frequency:", record.fs, "Hz")
print("Number of leads:", signal.shape[1])
print("Number of samples:", signal.shape[0])

# ============================================================
# 2. BASIC VALIDATION
# ============================================================

print()
print("--- RAW ECG VALIDATION ---")

print("NaN values:", np.isnan(signal).sum())
print("Infinite values:", np.isinf(signal).sum())

print("Raw minimum:", signal.min())
print("Raw maximum:", signal.max())
print("Raw mean:", signal.mean())
print("Raw standard deviation:", signal.std())

# ============================================================
# 3. BASELINE CORRECTION
# ============================================================
#
# For this controlled experiment we remove the mean of each
# lead. This is a simple baseline-centering operation.
#
# This is NOT intended to replace a clinical ECG filtering
# pipeline.
# ============================================================

baseline = np.mean(signal, axis=0)

centered_signal = signal - baseline

print()
print("--- BASELINE CENTERING ---")

print("Mean before centering:")

for i, value in enumerate(baseline):
    print(f"Lead {i+1:2d}: {value:.8f}")

print()
print("Mean after centering:")

for i in range(centered_signal.shape[1]):
    print(
        f"Lead {i+1:2d}: "
        f"{np.mean(centered_signal[:, i]):.8f}"
    )

# ============================================================
# 4. PER-LEAD STANDARDIZATION
# ============================================================
#
# Each lead is standardized independently:
#
#       X_standardized = (X - mean) / standard deviation
#
# This gives each lead approximately zero mean and unit
# standard deviation.
# ============================================================

lead_mean = np.mean(centered_signal, axis=0)

lead_std = np.std(centered_signal, axis=0)

# Prevent division by zero

lead_std[lead_std == 0] = 1.0

normalized_signal = (
    centered_signal - lead_mean
) / lead_std

print()
print("--- NORMALIZATION ---")

print("Normalized minimum:", normalized_signal.min())
print("Normalized maximum:", normalized_signal.max())
print("Normalized mean:", normalized_signal.mean())
print("Normalized standard deviation:", normalized_signal.std())

# ============================================================
# 5. FINAL SHAPE CHECK
# ============================================================

print()
print("--- MODEL INPUT CHECK ---")

print("Input shape:", normalized_signal.shape)

expected_shape = (1000, 12)

if normalized_signal.shape == expected_shape:

    print("Expected shape:", expected_shape)
    print("Shape check: PASS")

else:

    print("Expected shape:", expected_shape)
    print("Shape check: FAIL")

# ============================================================
# 6. NUMERICAL VALIDATION
# ============================================================

nan_count = np.isnan(normalized_signal).sum()

inf_count = np.isinf(normalized_signal).sum()

mean_error = np.max(
    np.abs(np.mean(normalized_signal, axis=0))
)

std_error = np.max(
    np.abs(np.std(normalized_signal, axis=0) - 1)
)

print()
print("--- NORMALIZED DATA VALIDATION ---")

print("NaN values:", nan_count)

print("Infinite values:", inf_count)

print(
    "Maximum absolute lead mean:",
    mean_error
)

print(
    "Maximum deviation from unit standard deviation:",
    std_error
)

if (
    nan_count == 0
    and inf_count == 0
    and mean_error < 1e-5
    and std_error < 1e-5
):

    validation_status = "PASS"

else:

    validation_status = "FAIL"

print("Normalization validation:", validation_status)

# ============================================================
# 7. PLOT RAW VS PREPROCESSED ECG
# ============================================================

time = np.arange(signal.shape[0]) / record.fs

fig, axes = plt.subplots(
    2,
    1,
    figsize=(12, 8)
)

# Lead I

axes[0].plot(
    time,
    signal[:, 0]
)

axes[0].set_title(
    "Raw PTB-XL ECG - Lead I"
)

axes[0].set_xlabel(
    "Time (seconds)"
)

axes[0].set_ylabel(
    "Amplitude (mV)"
)

axes[0].grid(True)

# Normalized Lead I

axes[1].plot(
    time,
    normalized_signal[:, 0]
)

axes[1].set_title(
    "Preprocessed PTB-XL ECG - Lead I"
)

axes[1].set_xlabel(
    "Time (seconds)"
)

axes[1].set_ylabel(
    "Normalized amplitude"
)

axes[1].grid(True)

plt.tight_layout()

plt.savefig(
    PLOT_FILE,
    dpi=150
)

plt.close()

# ============================================================
# 8. OVERALL RESULT
# ============================================================

overall_status = "PASS"

if validation_status != "PASS":
    overall_status = "FAIL"

print()
print("=" * 70)
print("OVERALL RESULT:", overall_status)
print("=" * 70)

print()
print("Plot saved to:")
print(PLOT_FILE)

# ============================================================
# 9. SAVE REPORT
# ============================================================

report = []

report.append(
    "EXPERIMENT 3A - PTB-XL ECG PREPROCESSING PIPELINE"
)

report.append("=" * 70)

report.append(
    f"Record: {record.record_name}"
)

report.append(
    f"Sampling frequency: {record.fs} Hz"
)

report.append(
    f"Number of leads: {signal.shape[1]}"
)

report.append(
    f"Number of samples: {signal.shape[0]}"
)

report.append(
    f"Raw shape: {signal.shape}"
)

report.append(
    f"Processed shape: {normalized_signal.shape}"
)

report.append(
    f"Raw NaN values: {np.isnan(signal).sum()}"
)

report.append(
    f"Raw infinite values: {np.isinf(signal).sum()}"
)

report.append(
    f"Processed NaN values: {nan_count}"
)

report.append(
    f"Processed infinite values: {inf_count}"
)

report.append(
    f"Maximum absolute normalized lead mean: {mean_error}"
)

report.append(
    f"Maximum deviation from unit standard deviation: {std_error}"
)

report.append(
    f"Normalization validation: {validation_status}"
)

report.append("")
report.append(
    "PREPROCESSING STEPS"
)

report.append(
    "1. Loaded PTB-XL ECG using WFDB."
)

report.append(
    "2. Validated raw ECG values."
)

report.append(
    "3. Centered each lead by subtracting its mean."
)

report.append(
    "4. Standardized each lead using its standard deviation."
)

report.append(
    "5. Verified numerical integrity."
)

report.append(
    "6. Verified model input dimensions."
)

report.append("")
report.append(
    f"OVERALL RESULT: {overall_status}"
)

REPORT_FILE.write_text(
    "\n".join(report),
    encoding="utf-8"
)

print()
print("Report saved to:")
print(REPORT_FILE)

print()
print("Experiment 3A completed.")