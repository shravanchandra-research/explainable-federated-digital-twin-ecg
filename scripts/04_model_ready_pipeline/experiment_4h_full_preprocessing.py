import os
import ast
import numpy as np
import pandas as pd
import wfdb

# ============================================================
# EXPERIMENT 4H
# FULL PTB-XL MODEL-READY DATASET GENERATION
# ============================================================

BASE_DIR = os.environ.get("PTBXL_ROOT", r"C:\PTB-XL")

DATABASE_FILE = os.path.join(
    BASE_DIR,
    "ptbxl_database.csv"
)

RECORDS_FILE = os.path.join(
    BASE_DIR,
    "RECORDS"
)

SOURCE_DIR = os.path.join(
    BASE_DIR,
    "records100"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "experiments",
    "4H_output"
)

PREPROCESSED_DIR = os.path.join(
    OUTPUT_DIR,
    "preprocessed"
)

INDEX_FILE = os.path.join(
    OUTPUT_DIR,
    "experiment_4h_full_index.csv"
)

ERROR_FILE = os.path.join(
    OUTPUT_DIR,
    "experiment_4h_errors.csv"
)

EXPECTED_SHAPE = (1000, 12)

TARGET_CLASSES = [
    "NORM",
    "MI",
    "STTC",
    "CD",
    "HYP"
]

# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

os.makedirs(
    PREPROCESSED_DIR,
    exist_ok=True
)

# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("EXPERIMENT 4H")
print("FULL PTB-XL MODEL-READY DATASET GENERATION")
print("=" * 70)

# ============================================================
# LOAD PTB-XL DATABASE
# ============================================================

print("\nLoading PTB-XL metadata...")

df = pd.read_csv(
    DATABASE_FILE
)

print(
    "Metadata records:",
    len(df)
)

required_columns = [
    "ecg_id",
    "patient_id",
    "filename_lr",
    "strat_fold",
    "scp_codes"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    print("\nERROR: Missing metadata columns:")
    print(missing_columns)

    raise SystemExit(1)

print(
    "Required metadata columns: PASS"
)

# ============================================================
# LOAD RECORDS FILE
# ============================================================

print("\nLoading RECORDS file...")

with open(
    RECORDS_FILE,
    "r",
    encoding="utf-8"
) as f:

    record_lines = [
        line.strip()
        for line in f
        if line.strip()
    ]

lr_records = [
    line
    for line in record_lines
    if line.endswith("_lr")
]

print(
    "Total RECORDS entries:",
    len(record_lines)
)

print(
    "Low-resolution records:",
    len(lr_records)
)

if len(lr_records) != 21798:

    print("\nWARNING:")
    print(
        "Expected 21,798 low-resolution entries."
    )
    print(
        "Detected:",
        len(lr_records)
    )

# ============================================================
# LOAD SCP DIAGNOSTIC MAPPINGS
# ============================================================

print("\nLoading SCP diagnostic mappings...")

SCP_FILE = os.path.join(
    BASE_DIR,
    "scp_statements.csv"
)

scp_df = pd.read_csv(
    SCP_FILE,
    index_col=0
)

if "diagnostic_class" not in scp_df.columns:

    print(
        "\nERROR: diagnostic_class column "
        "not found in scp_statements.csv"
    )

    raise SystemExit(1)

scp_to_class = {}

for code, row in scp_df.iterrows():

    diagnostic_class = row[
        "diagnostic_class"
    ]

    if pd.notna(
        diagnostic_class
    ):

        diagnostic_class = str(
            diagnostic_class
        ).strip()

        if diagnostic_class in TARGET_CLASSES:

            scp_to_class[
                str(code)
            ] = diagnostic_class

print(
    "Diagnostic SCP mappings:",
    len(scp_to_class)
)

# ============================================================
# PREPROCESSING FUNCTION
# ============================================================

def preprocess_ecg(signal):

    signal = np.asarray(
        signal,
        dtype=np.float64
    )

    # --------------------------------------------------------
    # Shape validation
    # --------------------------------------------------------

    if signal.shape != EXPECTED_SHAPE:

        raise ValueError(
            "Invalid shape: "
            + str(signal.shape)
        )

    # --------------------------------------------------------
    # NaN / Infinite validation
    # --------------------------------------------------------

    if not np.isfinite(
        signal
    ).all():

        raise ValueError(
            "Signal contains NaN or infinite values"
        )

    # --------------------------------------------------------
    # Per-lead baseline removal
    # --------------------------------------------------------

    lead_means = np.mean(
        signal,
        axis=0,
        keepdims=True
    )

    centered = (
        signal - lead_means
    )

    # --------------------------------------------------------
    # Per-lead standard deviation
    # --------------------------------------------------------

    lead_std = np.std(
        centered,
        axis=0,
        keepdims=True
    )

    if np.any(
        lead_std <= 0
    ):

        raise ValueError(
            "Zero standard deviation detected"
        )

    # --------------------------------------------------------
    # Per-lead z-score normalization
    # --------------------------------------------------------

    normalized = (
        centered / lead_std
    )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    if not np.isfinite(
        normalized
    ).all():

        raise ValueError(
            "Normalized signal contains "
            "NaN or infinite values"
        )

    return normalized.astype(
        np.float32
    )


# ============================================================
# CHECK EXISTING OUTPUTS
# ============================================================

print(
    "\nChecking existing preprocessed files..."
)

existing_valid = 0
existing_invalid = 0

if os.path.exists(
    PREPROCESSED_DIR
):

    for root, dirs, files in os.walk(
        PREPROCESSED_DIR
    ):

        for filename in files:

            if not filename.endswith(
                ".npy"
            ):
                continue

            filepath = os.path.join(
                root,
                filename
            )

            try:

                arr = np.load(
                    filepath,
                    mmap_mode="r"
                )

                if (
                    arr.shape == EXPECTED_SHAPE
                    and np.isfinite(arr).all()
                ):

                    existing_valid += 1

                else:

                    existing_invalid += 1

            except Exception:

                existing_invalid += 1

print(
    "Existing valid outputs:",
    existing_valid
)

print(
    "Existing invalid outputs:",
    existing_invalid
)

# ============================================================
# SORT DATABASE
# ============================================================

df = df.sort_values(
    "ecg_id"
).reset_index(
    drop=True
)

total_records = len(df)

# ============================================================
# PROCESS ALL RECORDS
# ============================================================

print(
    "\nStarting full preprocessing..."
)

print(
    "Automatic resume: ENABLED"
)

index_records = []
error_records = []

processed_count = 0
skipped_count = 0
failed_count = 0

shape_errors = 0
nan_errors = 0
normalization_errors = 0
label_errors = 0

# ============================================================
# MAIN LOOP
# ============================================================

for position, (_, row) in enumerate(
    df.iterrows(),
    start=1
):

    ecg_id = int(
        row["ecg_id"]
    )

    patient_id = row[
        "patient_id"
    ]

    filename_lr = str(
        row["filename_lr"]
    ).replace(
        "\\",
        "/"
    )

    record_name = os.path.basename(
        filename_lr
    )

    # --------------------------------------------------------
    # IMPORTANT PATH FIX
    #
    # filename_lr contains:
    # records100/00000/00001_lr
    #
    # SOURCE_DIR already points to:
    # C:\PTB-XL\records100
    #
    # Therefore remove "records100/" first.
    # --------------------------------------------------------

    relative_filename = filename_lr

    if relative_filename.startswith(
        "records100/"
    ):

        relative_filename = (
            relative_filename[
                len("records100/") :
            ]
        )

    # --------------------------------------------------------
    # OUTPUT DIRECTORY
    # --------------------------------------------------------

    relative_directory = os.path.dirname(
        relative_filename
    )

    output_directory = os.path.join(
        PREPROCESSED_DIR,
        relative_directory
    )

    os.makedirs(
        output_directory,
        exist_ok=True
    )

    output_file = os.path.join(
        output_directory,
        record_name + ".npy"
    )

    # --------------------------------------------------------
    # FOLD
    # --------------------------------------------------------

    fold = int(
        row["strat_fold"]
    )

    # --------------------------------------------------------
    # SPLIT
    # --------------------------------------------------------

    if fold in range(1, 9):

        split = "train"

    elif fold == 9:

        split = "validation"

    elif fold == 10:

        split = "test"

    else:

        split = "invalid"

    # --------------------------------------------------------
    # LABELS
    # --------------------------------------------------------

    target_vector = {
        target: 0
        for target in TARGET_CLASSES
    }

    try:

        scp_codes = ast.literal_eval(
            str(row["scp_codes"])
        )

        if not isinstance(
            scp_codes,
            dict
        ):

            raise ValueError(
                "scp_codes is not a dictionary"
            )

        for code in scp_codes:

            code = str(
                code
            ).strip()

            if code in scp_to_class:

                target_class = scp_to_class[
                    code
                ]

                target_vector[
                    target_class
                ] = 1

    except Exception as e:

        label_errors += 1
        failed_count += 1

        error_records.append({
            "ecg_id": ecg_id,
            "record_name": record_name,
            "error_type": "LABEL_ERROR",
            "error": str(e)
        })

        continue

    # --------------------------------------------------------
    # RESUME CHECK
    # --------------------------------------------------------

    if os.path.exists(
        output_file
    ):

        try:

            existing = np.load(
                output_file,
                mmap_mode="r"
            )

            if (
                existing.shape == EXPECTED_SHAPE
                and np.isfinite(existing).all()
            ):

                skipped_count += 1

                index_records.append({

                    "ecg_id": ecg_id,

                    "patient_id": patient_id,

                    "record_name": record_name,

                    "filename_lr": filename_lr,

                    "processed_file": output_file,

                    "fold": fold,

                    "split": split,

                    "NORM":
                        target_vector["NORM"],

                    "MI":
                        target_vector["MI"],

                    "STTC":
                        target_vector["STTC"],

                    "CD":
                        target_vector["CD"],

                    "HYP":
                        target_vector["HYP"],

                    "samples": 1000,

                    "leads": 12,

                    "status":
                        "SKIPPED_VALID"
                })

                if (
                    position % 100 == 0
                    or position == total_records
                ):

                    print(
                        f"[{position}/{total_records}] "
                        f"Processed={processed_count} "
                        f"Skipped={skipped_count} "
                        f"Failed={failed_count}"
                    )

                continue

        except Exception:

            # Invalid existing output.
            # It will be regenerated below.

            pass

    # --------------------------------------------------------
    # SOURCE RECORD PATH
    # --------------------------------------------------------

    source_record = os.path.join(
        SOURCE_DIR,
        relative_filename
    )

    # WFDB requires path without .dat/.hea

    if source_record.endswith(
        ".dat"
    ):

        source_record = source_record[
            :-4
        ]

    # --------------------------------------------------------
    # LOAD AND PROCESS
    # --------------------------------------------------------

    try:

        record = wfdb.rdrecord(
            source_record
        )

        signal = record.p_signal

        # ----------------------------------------------------
        # PREPROCESS
        # ----------------------------------------------------

        try:

            processed = preprocess_ecg(
                signal
            )

        except ValueError as e:

            message = str(e)

            if "shape" in message.lower():

                shape_errors += 1

            elif (
                "NaN" in message
                or "infinite" in message
            ):

                nan_errors += 1

            else:

                normalization_errors += 1

            raise

        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        np.save(
            output_file,
            processed
        )

        # ----------------------------------------------------
        # VALIDATE SAVED FILE
        # ----------------------------------------------------

        validation = np.load(
            output_file
        )

        if validation.shape != EXPECTED_SHAPE:

            raise ValueError(
                "Saved output shape invalid: "
                + str(validation.shape)
            )

        if not np.isfinite(
            validation
        ).all():

            raise ValueError(
                "Saved output contains "
                "NaN or infinite values"
            )

        # ----------------------------------------------------
        # ADD TO INDEX
        # ----------------------------------------------------

        index_records.append({

            "ecg_id": ecg_id,

            "patient_id": patient_id,

            "record_name": record_name,

            "filename_lr": filename_lr,

            "processed_file": output_file,

            "fold": fold,

            "split": split,

            "NORM":
                target_vector["NORM"],

            "MI":
                target_vector["MI"],

            "STTC":
                target_vector["STTC"],

            "CD":
                target_vector["CD"],

            "HYP":
                target_vector["HYP"],

            "samples": 1000,

            "leads": 12,

            "status":
                "PROCESSED"
        })

        processed_count += 1

    except Exception as e:

        failed_count += 1

        error_records.append({

            "ecg_id": ecg_id,

            "record_name": record_name,

            "error_type":
                "PROCESSING_ERROR",

            "error": str(e)
        })

    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    if (
        position % 100 == 0
        or position == total_records
    ):

        print(
            f"[{position}/{total_records}] "
            f"Processed={processed_count} "
            f"Skipped={skipped_count} "
            f"Failed={failed_count}"
        )

# ============================================================
# EMPTY INDEX PROTECTION
# ============================================================

if len(index_records) == 0:

    print(
        "\nERROR: No records were successfully "
        "processed or recovered."
    )

    error_df = pd.DataFrame(
        error_records
    )

    error_df.to_csv(
        ERROR_FILE,
        index=False
    )

    print(
        "\nError report:"
    )

    print(
        ERROR_FILE
    )

    raise SystemExit(1)

# ============================================================
# CREATE INDEX
# ============================================================

print(
    "\nCreating full model-ready index..."
)

index_df = pd.DataFrame(
    index_records
)

index_df = index_df.sort_values(
    "ecg_id"
)

index_df.to_csv(
    INDEX_FILE,
    index=False
)

# ============================================================
# ERROR REPORT
# ============================================================

error_df = pd.DataFrame(
    error_records
)

error_df.to_csv(
    ERROR_FILE,
    index=False
)

# ============================================================
# FINAL VALIDATION
# ============================================================

print(
    "\nPerforming final dataset validation..."
)

final_index = pd.read_csv(
    INDEX_FILE
)

records_in_index = len(
    final_index
)

duplicate_ecg = int(
    final_index[
        "ecg_id"
    ].duplicated().sum()
)

missing_processed = int(
    (
        ~final_index[
            "processed_file"
        ].apply(
            os.path.exists
        )
    ).sum()
)

invalid_shapes = 0
invalid_finite = 0

for filepath in final_index[
    "processed_file"
]:

    try:

        arr = np.load(
            filepath,
            mmap_mode="r"
        )

        if arr.shape != EXPECTED_SHAPE:

            invalid_shapes += 1

        if not np.isfinite(
            arr
        ).all():

            invalid_finite += 1

    except Exception:

        invalid_shapes += 1

# ============================================================
# SPLIT COUNTS
# ============================================================

train_count = int(
    (
        final_index["split"]
        == "train"
    ).sum()
)

validation_count = int(
    (
        final_index["split"]
        == "validation"
    ).sum()
)

test_count = int(
    (
        final_index["split"]
        == "test"
    ).sum()
)

# ============================================================
# CLASS COUNTS
# ============================================================

class_counts = {}

for target in TARGET_CLASSES:

    class_counts[target] = int(
        final_index[target].sum()
    )

# ============================================================
# FINAL SUMMARY
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "EXPERIMENT 4H FINAL SUMMARY"
)

print(
    "=" * 70
)

print(
    "Metadata records              :",