import os
import wfdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# EXPERIMENT 1A
# PTB-XL ECG Waveform Verification
# ============================================================

BASE_DIR = os.environ.get("PTBXL_ROOT", r"C:\PTB-XL")
RECORD_PATH = os.path.join(
    BASE_DIR,
    "records100",
    "01000",
    "01000_lr"
)

OUTPUT_DIR = os.path.join(BASE_DIR, "experiments")
REPORT_FILE = os.path.join(
    OUTPUT_DIR,
    "experiment_1a_verification.txt"
)

PLOT_FILE = os.path.join(
    OUTPUT_DIR,
    "experiment_1a_12lead_ecg.png"
)

# ------------------------------------------------------------
# Create output directory
# ------------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------
# Read ECG using WFDB
# ------------------------------------------------------------

print("=" * 70)
print("EXPERIMENT 1A - PTB-XL ECG WAVEFORM VERIFICATION")
print("=" * 70)

print("\nReading record:")
print(RECORD_PATH)

record = wfdb.rdrecord(RECORD_PATH)

signals = record.p_signal
fields = record.__dict__

# ------------------------------------------------------------
# Basic information
# ------------------------------------------------------------

num_samples = signals.shape[0]
num_channels = signals.shape[1]
sampling_frequency = record.fs
duration_seconds = num_samples / sampling_frequency

lead_names = record.sig_name

# ------------------------------------------------------------
# Signal quality checks
# ------------------------------------------------------------

nan_count = np.isnan(signals).sum()
inf_count = np.isinf(signals).sum()

minimum_value = np.min(signals)
maximum_value = np.max(signals)
mean_value = np.mean(signals)
std_value = np.std(signals)

# ------------------------------------------------------------
# Display results
# ------------------------------------------------------------

print("\n--- RECORD INFORMATION ---")

print("Record name       :", record.record_name)
print("Number of leads   :", num_channels)
print("Samples per lead  :", num_samples)
print("Sampling rate     :", sampling_frequency, "Hz")
print("Duration          :", duration_seconds, "seconds")

print("\n--- LEADS ---")

for i, lead in enumerate(lead_names, start=1):
    print(f"{i:2d}. {lead}")

print("\n--- SIGNAL VALIDATION ---")

print("NaN values        :", nan_count)
print("Infinite values   :", inf_count)
print("Minimum value     :", minimum_value)
print("Maximum value     :", maximum_value)
print("Mean value        :", mean_value)
print("Standard deviation:", std_value)

# ------------------------------------------------------------
# Expected structure
# ------------------------------------------------------------

expected_leads = [
    "I", "II", "III",
    "AVR", "AVL", "AVF",
    "V1", "V2", "V3",
    "V4", "V5", "V6"
]

print("\n--- STRUCTURAL VERIFICATION ---")

checks = {}

checks["12 leads"] = num_channels == 12
checks["100 Hz"] = sampling_frequency == 100
checks["1000 samples"] = num_samples == 1000
checks["10 seconds"] = abs(duration_seconds - 10.0) < 1e-9
checks["Expected lead names"] = lead_names == expected_leads
checks["No NaN"] = nan_count == 0
checks["No Inf"] = inf_count == 0

for check_name, result in checks.items():
    print(f"{check_name:25s}: {'PASS' if result else 'FAIL'}")

overall_result = all(checks.values())

print("\nOVERALL VERIFICATION:",
      "PASS" if overall_result else "FAIL")

# ------------------------------------------------------------
# Generate ECG plot
# ------------------------------------------------------------

time_axis = np.arange(num_samples) / sampling_frequency

fig, axes = plt.subplots(
    12,
    1,
    figsize=(14, 18),
    sharex=True
)

for i in range(12):

    axes[i].plot(
        time_axis,
        signals[:, i],
        linewidth=0.8
    )

    axes[i].set_ylabel(
        lead_names[i],
        rotation=0,
        labelpad=25
    )

    axes[i].grid(True, alpha=0.3)

axes[-1].set_xlabel("Time (seconds)")

fig.suptitle(
    "PTB-XL Record 01000_lr - 12 Lead ECG",
    fontsize=16
)

plt.tight_layout()

plt.savefig(
    PLOT_FILE,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

# ------------------------------------------------------------
# Generate text report
# ------------------------------------------------------------

with open(REPORT_FILE, "w", encoding="utf-8") as f:

    f.write("EXPERIMENT 1A - PTB-XL ECG WAVEFORM VERIFICATION\n")
    f.write("=" * 60 + "\n\n")

    f.write(f"Record name: {record.record_name}\n")
    f.write(f"Number of leads: {num_channels}\n")
    f.write(f"Samples per lead: {num_samples}\n")
    f.write(f"Sampling frequency: {sampling_frequency} Hz\n")
    f.write(f"Duration: {duration_seconds} seconds\n")

    f.write("\nLead names:\n")

    for lead in lead_names:
        f.write(f"- {lead}\n")

    f.write("\nSignal validation:\n")
    f.write(f"NaN values: {nan_count}\n")
    f.write(f"Infinite values: {inf_count}\n")
    f.write(f"Minimum value: {minimum_value}\n")
    f.write(f"Maximum value: {maximum_value}\n")
    f.write(f"Mean value: {mean_value}\n")
    f.write(f"Standard deviation: {std_value}\n")

    f.write("\nStructural verification:\n")

    for check_name, result in checks.items():
        f.write(
            f"{check_name}: "
            f"{'PASS' if result else 'FAIL'}\n"
        )

    f.write(
        "\nOVERALL VERIFICATION: "
        + ("PASS" if overall_result else "FAIL")
        + "\n"
    )

print("\n--- OUTPUT FILES ---")

print("Report:")
print(REPORT_FILE)

print("\nECG plot:")
print(PLOT_FILE)

print("\nExperiment 1A completed.")