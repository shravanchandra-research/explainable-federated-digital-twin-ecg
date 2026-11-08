import os
import csv
import numpy as np
import wfdb

# ============================================================
# EXPERIMENT 4B
# PTB-XL BATCH PREPROCESSING VALIDATION
# ============================================================

ROOT = os.environ.get("PTBXL_ROOT", r"C:\PTB-XL")
RECORDS_DIR = os.path.join(ROOT, "records100")
OUTPUT_DIR = os.path.join(ROOT, "experiments", "4B_output")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------
# Number of records to validate
# ------------------------------------------------------------

N_RECORDS = 100

# ------------------------------------------------------------
# Select first N records from RECORDS
# ------------------------------------------------------------

records_file = os.path.join(ROOT, "RECORDS")

all_records = []

with open(records_file, "r") as f:
    for line in f:
        line = line.strip()

        if line.startswith("records100/") and line.endswith("_lr"):
            all_records.append(line)

all_records = all_records[:N_RECORDS]

print("=" * 70)
print("EXPERIMENT 4B - PTB-XL BATCH PREPROCESSING VALIDATION")
print("=" * 70)

print("\nRecords available in RECORDS100 :", len(all_records))
print("Records selected for validation :", len(all_records))

# ------------------------------------------------------------
# Statistics
# ------------------------------------------------------------

successful = 0
failed = 0

nan_total = 0
inf_total = 0

shape_errors = 0
normalization_errors = 0

mean_errors = 0
std_errors = 0

results = []

# ------------------------------------------------------------
# Process records
# ------------------------------------------------------------

for index, relative_record in enumerate(all_records, start=1):

    record_name = os.path.basename(relative_record)
    record_path = os.path.join(
        ROOT,
        relative_record.replace("/", os.sep)
    )

    print(
        "\n[{}/{}] Processing {}".format(
            index,
            len(all_records),
            relative_record
        )
    )

    try:

        # ----------------------------------------------------
        # Load ECG
        # ----------------------------------------------------

        record = wfdb.rdrecord(record_path)

        signal = record.p_signal.astype(np.float64)

        # ----------------------------------------------------
        # Basic validation
        # ----------------------------------------------------

        expected_shape = (1000, 12)

        if signal.shape != expected_shape:
            shape_errors += 1

            print(
                "  ERROR: Unexpected shape:",
                signal.shape
            )

            failed += 1
            continue

        # ----------------------------------------------------
        # NaN / Inf check
        # ----------------------------------------------------

        nan_count = int(np.isnan(signal).sum())
        inf_count = int(np.isinf(signal).sum())

        nan_total += nan_count
        inf_total += inf_count

        if nan_count > 0 or inf_count > 0:

            print(
                "  ERROR: NaN =", nan_count,
                "Inf =", inf_count
            )

            failed += 1
            continue

        # ----------------------------------------------------
        # Baseline removal
        # ----------------------------------------------------

        lead_mean = np.mean(signal, axis=0)

        centered_signal = signal - lead_mean

        # ----------------------------------------------------
        # Calculate standard deviation
        # ----------------------------------------------------

        lead_std = np.std(centered_signal, axis=0)

        if np.any(lead_std == 0):

            print("  ERROR: Zero standard deviation detected.")

            normalization_errors += 1
            failed += 1
            continue

        # ----------------------------------------------------
        # Z-score normalization
        # ----------------------------------------------------

        normalized_signal = centered_signal / lead_std

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        normalized_mean = np.mean(
            normalized_signal,
            axis=0
        )

        normalized_std = np.std(
            normalized_signal,
            axis=0
        )

        max_mean_error = float(
            np.max(np.abs(normalized_mean))
        )

        max_std_error = float(
            np.max(np.abs(normalized_std - 1.0))
        )

        # ----------------------------------------------------
        # Numerical tolerance
        # ----------------------------------------------------

        mean_ok = max_mean_error < 1e-10
        std_ok = max_std_error < 1e-10

        if not mean_ok:
            mean_errors += 1

        if not std_ok:
            std_errors += 1

        if not mean_ok or not std_ok:
            normalization_errors += 1
            failed += 1

            print("  ERROR: Normalization validation failed.")

            continue

        # ----------------------------------------------------
        # Successful record
        # ----------------------------------------------------

        successful += 1

        results.append([
            record_name,
            record.fs,
            record.n_sig,
            record.sig_len,
            nan_count,
            inf_count,
            max_mean_error,
            max_std_error,
            float(np.min(normalized_signal)),
            float(np.max(normalized_signal))
        ])

        print("  Shape              :", signal.shape)
        print("  NaN                :", nan_count)
        print("  Infinite           :", inf_count)
        print("  Maximum mean error :", max_mean_error)
        print("  Maximum std error  :", max_std_error)
        print("  Status             : PASS")

    except Exception as e:

        failed += 1

        print("  ERROR:", str(e))

# ------------------------------------------------------------
# Save CSV report
# ------------------------------------------------------------

csv_file = os.path.join(
    OUTPUT_DIR,
    "experiment_4b_validation.csv"
)

with open(
    csv_file,
    "w",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "record",
        "sampling_frequency",
        "leads",
        "samples",
        "nan_count",
        "inf_count",
        "max_normalized_mean_error",
        "max_normalized_std_error",
        "normalized_min",
        "normalized_max"
    ])

    writer.writerows(results)

# ------------------------------------------------------------
# Final summary
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("EXPERIMENT 4B SUMMARY")
print("=" * 70)

print("Records selected        :", len(all_records))
print("Successful records      :", successful)
print("Failed records          :", failed)

print("Total NaN values        :", nan_total)
print("Total infinite values  :", inf_total)

print("Shape errors            :", shape_errors)
print("Normalization errors   :", normalization_errors)

print("Mean validation errors :", mean_errors)
print("STD validation errors  :", std_errors)

print("\nValidation report:")
print(csv_file)

# ------------------------------------------------------------
# Overall result
# ------------------------------------------------------------

if (
    successful == len(all_records)
    and failed == 0
    and nan_total == 0
    and inf_total == 0
    and shape_errors == 0
    and normalization_errors == 0
):

    print("\n" + "=" * 70)
    print("EXPERIMENT 4B RESULT : PASS")
    print("=" * 70)

else:

    print("\n" + "=" * 70)
    print("EXPERIMENT 4B RESULT : REVIEW REQUIRED")
    print("=" * 70)