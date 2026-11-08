import os
import csv
import numpy as np
import wfdb

# ============================================================
# EXPERIMENT 4C
# PTB-XL PREPROCESSED DATASET GENERATION
# ============================================================

ROOT = os.environ.get("PTBXL_ROOT", r"C:\PTB-XL")

RECORDS_DIR = os.path.join(
    ROOT,
    "records100"
)

OUTPUT_DIR = os.path.join(
    ROOT,
    "experiments",
    "4C_output"
)

PROCESSED_DIR = os.path.join(
    OUTPUT_DIR,
    "preprocessed"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

os.makedirs(
    PROCESSED_DIR,
    exist_ok=True
)

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

# Number of records to process
# Set to 100 for initial validation.
# After successful validation, change to 1000 or 21798.
N_RECORDS = 100

# ------------------------------------------------------------
# Input RECORDS file
# ------------------------------------------------------------

records_file = os.path.join(
    ROOT,
    "RECORDS"
)

# ------------------------------------------------------------
# Read records100 entries
# ------------------------------------------------------------

all_records = []

with open(
    records_file,
    "r"
) as f:

    for line in f:

        line = line.strip()

        if (
            line.startswith("records100/")
            and line.endswith("_lr")
        ):

            all_records.append(line)

# ------------------------------------------------------------
# Select requested number
# ------------------------------------------------------------

all_records = all_records[:N_RECORDS]

# ------------------------------------------------------------
# Output index
# ------------------------------------------------------------

index_file = os.path.join(
    OUTPUT_DIR,
    "experiment_4c_index.csv"
)

# ------------------------------------------------------------
# Counters
# ------------------------------------------------------------

successful = 0
skipped = 0
failed = 0

shape_errors = 0
nan_errors = 0
inf_errors = 0
normalization_errors = 0

# ------------------------------------------------------------
# Results
# ------------------------------------------------------------

results = []

# ------------------------------------------------------------
# Header
# ------------------------------------------------------------

print("=" * 70)
print("EXPERIMENT 4C - PTB-XL PREPROCESSED DATASET GENERATION")
print("=" * 70)

print()
print(
    "Records available in RECORDS100 :",
    len(all_records)
)

print(
    "Records selected for processing :",
    len(all_records)
)

print()
print("Output directory:")
print(PROCESSED_DIR)

# ------------------------------------------------------------
# Process each ECG
# ------------------------------------------------------------

for index, relative_record in enumerate(
    all_records,
    start=1
):

    record_name = os.path.basename(
        relative_record
    )

    record_path = os.path.join(
        ROOT,
        relative_record.replace(
            "/",
            os.sep
        )
    )

    # --------------------------------------------------------
    # Determine output path
    # --------------------------------------------------------

    parts = relative_record.split("/")

    # Example:
    # records100/00000/00001_lr

    folder_name = parts[-2]

    output_folder = os.path.join(
        PROCESSED_DIR,
        folder_name
    )

    os.makedirs(
        output_folder,
        exist_ok=True
    )

    output_file = os.path.join(
        output_folder,
        record_name + ".npy"
    )

    print()
    print(
        "[{}/{}] Processing {}".format(
            index,
            len(all_records),
            relative_record
        )
    )

    # --------------------------------------------------------
    # Resume support
    # --------------------------------------------------------

    if os.path.exists(output_file):

        try:

            existing = np.load(
                output_file
            )

            if (
                existing.shape == (1000, 12)
                and np.all(np.isfinite(existing))
            ):

                print(
                    "  Existing valid output found - SKIPPED"
                )

                skipped += 1

                results.append([
                    record_name,
                    relative_record,
                    record.fs if hasattr(record, "fs") else 100,
                    12,
                    1000,
                    "SKIPPED",
                    output_file
                ])

                continue

        except Exception:

            print(
                "  Existing output invalid - regenerating"
            )

    # --------------------------------------------------------
    # Load ECG
    # --------------------------------------------------------

    try:

        record = wfdb.rdrecord(
            record_path
        )

        signal = record.p_signal.astype(
            np.float64
        )

        # ----------------------------------------------------
        # Shape validation
        # ----------------------------------------------------

        if signal.shape != (1000, 12):

            print(
                "  ERROR: Unexpected shape:",
                signal.shape
            )

            shape_errors += 1
            failed += 1
            continue

        # ----------------------------------------------------
        # NaN validation
        # ----------------------------------------------------

        nan_count = int(
            np.isnan(signal).sum()
        )

        if nan_count > 0:

            print(
                "  ERROR: NaN values:",
                nan_count
            )

            nan_errors += 1
            failed += 1
            continue

        # ----------------------------------------------------
        # Infinite validation
        # ----------------------------------------------------

        inf_count = int(
            np.isinf(signal).sum()
        )

        if inf_count > 0:

            print(
                "  ERROR: Infinite values:",
                inf_count
            )

            inf_errors += 1
            failed += 1
            continue

        # ----------------------------------------------------
        # Per-lead baseline removal
        # ----------------------------------------------------

        lead_mean = np.mean(
            signal,
            axis=0
        )

        centered_signal = (
            signal - lead_mean
        )

        # ----------------------------------------------------
        # Per-lead standard deviation
        # ----------------------------------------------------

        lead_std = np.std(
            centered_signal,
            axis=0
        )

        # ----------------------------------------------------
        # Zero standard deviation check
        # ----------------------------------------------------

        if np.any(lead_std == 0):

            print(
                "  ERROR: Zero standard deviation detected"
            )

            normalization_errors += 1
            failed += 1
            continue

        # ----------------------------------------------------
        # Z-score normalization
        # ----------------------------------------------------

        normalized_signal = (
            centered_signal / lead_std
        )

        # ----------------------------------------------------
        # Final validation
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
            np.max(
                np.abs(normalized_mean)
            )
        )

        max_std_error = float(
            np.max(
                np.abs(
                    normalized_std - 1.0
                )
            )
        )

        if (
            max_mean_error >= 1e-10
            or max_std_error >= 1e-10
        ):

            print(
                "  ERROR: Normalization validation failed"
            )

            normalization_errors += 1
            failed += 1
            continue

        # ----------------------------------------------------
        # Save processed ECG
        # ----------------------------------------------------

        np.save(
            output_file,
            normalized_signal.astype(
                np.float32
            )
        )

        # ----------------------------------------------------
        # Verify saved file
        # ----------------------------------------------------

        saved_signal = np.load(
            output_file
        )

        if saved_signal.shape != (1000, 12):

            print(
                "  ERROR: Saved shape:",
                saved_signal.shape
            )

            shape_errors += 1
            failed += 1
            continue

        if not np.all(
            np.isfinite(saved_signal)
        ):

            print(
                "  ERROR: Saved file contains NaN/Inf"
            )

            failed += 1
            continue

        # ----------------------------------------------------
        # Successful record
        # ----------------------------------------------------

        successful += 1

        results.append([
            record_name,
            relative_record,
            record.fs,
            record.n_sig,
            record.sig_len,
            "PROCESSED",
            output_file
        ])

        print(
            "  Shape          :",
            signal.shape
        )

        print(
            "  Sampling rate  :",
            record.fs
        )

        print(
            "  Leads          :",
            record.n_sig
        )

        print(
            "  Samples        :",
            record.sig_len
        )

        print(
            "  Max mean error :",
            max_mean_error
        )

        print(
            "  Max std error  :",
            max_std_error
        )

        print(
            "  Saved          :",
            output_file
        )

        print(
            "  Status         : PASS"
        )

    except Exception as e:

        print(
            "  ERROR:",
            str(e)
        )

        failed += 1


# ============================================================
# SAVE INDEX CSV
# ============================================================

with open(
    index_file,
    "w",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "record",
        "source_record",
        "sampling_frequency",
        "leads",
        "samples",
        "status",
        "processed_file"
    ])

    writer.writerows(
        results
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("EXPERIMENT 4C SUMMARY")
print("=" * 70)

print(
    "Records selected        :",
    len(all_records)
)

print(
    "Successfully processed  :",
    successful
)

print(
    "Skipped valid outputs   :",
    skipped
)

print(
    "Failed records          :",
    failed
)

print(
    "Shape errors            :",
    shape_errors
)

print(
    "NaN errors              :",
    nan_errors
)

print(
    "Infinite errors         :",
    inf_errors
)

print(
    "Normalization errors    :",
    normalization_errors
)

print()
print(
    "Processed dataset:"
)

print(
    PROCESSED_DIR
)

print()
print(
    "Index file:"
)

print(
    index_file
)


# ============================================================
# FINAL RESULT
# ============================================================

total_valid = successful + skipped

if (
    total_valid == len(all_records)
    and failed == 0
    and shape_errors == 0
    and nan_errors == 0
    and inf_errors == 0
    and normalization_errors == 0
):

    print()
    print("=" * 70)
    print("EXPERIMENT 4C RESULT : PASS")
    print("=" * 70)

else:

    print()
    print("=" * 70)
    print("EXPERIMENT 4C RESULT : REVIEW REQUIRED")
    print("=" * 70)