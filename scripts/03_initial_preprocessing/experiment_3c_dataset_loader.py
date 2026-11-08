import os
import pandas as pd
import numpy as np
import wfdb
import ast

from pathlib import Path

# ============================================================
# EXPERIMENT 3C
# PTB-XL DATASET LOADER AND SPLIT VALIDATION
# CORRECTED VERSION
# ============================================================

BASE_DIR = Path(os.environ.get("PTBXL_ROOT", r"C:\PTB-XL"))

DB_FILE = BASE_DIR / "ptbxl_database.csv"
SCP_FILE = BASE_DIR / "scp_statements.csv"

RECORDS_DIR = BASE_DIR / "records100"

OUT_FILE = (
    BASE_DIR
    / "experiments"
    / "experiment_3c_dataset_loader.txt"
)

# ============================================================
# CONFIGURATION
# ============================================================

SUPERCLASSES = [
    "NORM",
    "MI",
    "STTC",
    "CD",
    "HYP"
]

TRAIN_FOLDS = list(range(1, 9))
VALIDATION_FOLD = 9
TEST_FOLD = 10

# The waveform that has actually been verified/downloaded
VERIFIED_RECORD = "01000_lr"

print("=" * 70)
print("EXPERIMENT 3C - PTB-XL DATASET LOADER AND SPLIT VALIDATION")
print("=" * 70)

# ============================================================
# 1. LOAD METADATA
# ============================================================

print()
print("--- LOADING METADATA ---")

df = pd.read_csv(DB_FILE)
scp = pd.read_csv(SCP_FILE)

print("Total ECG records :", len(df))
print("Unique patients   :", df["patient_id"].nunique())

# ============================================================
# 2. BUILD SCP -> SUPERCLASS MAPPING
# ============================================================

diagnostic_flag = (
    pd.to_numeric(
        scp["diagnostic"],
        errors="coerce"
    )
    .fillna(0)
)

diagnostic_scp = scp[
    diagnostic_flag == 1
].copy()

code_to_class = {}

for _, row in diagnostic_scp.iterrows():

    code = str(
        row["Unnamed: 0"]
    ).strip()

    diagnostic_class = row[
        "diagnostic_class"
    ]

    if pd.notna(diagnostic_class):

        diagnostic_class = str(
            diagnostic_class
        ).strip()

        if diagnostic_class in SUPERCLASSES:

            code_to_class[code] = (
                diagnostic_class
            )

print(
    "Mapped diagnostic codes:",
    len(code_to_class)
)

# ============================================================
# 3. PARSE SCP CODES
# ============================================================

def parse_scp_codes(value):

    if pd.isna(value):
        return {}

    try:

        result = ast.literal_eval(
            str(value)
        )

        if isinstance(result, dict):
            return result

        return {}

    except Exception:

        return {}


# ============================================================
# 4. CREATE TARGET VECTOR
# ============================================================

def create_target_vector(scp_value):

    codes = parse_scp_codes(
        scp_value
    )

    classes = set()

    for code in codes.keys():

        code = str(code).strip()

        if code in code_to_class:

            classes.add(
                code_to_class[code]
            )

    vector = np.array(
        [
            1 if label in classes else 0
            for label in SUPERCLASSES
        ],
        dtype=np.float32
    )

    return vector


# ============================================================
# 5. PREPROCESS ECG
# ============================================================

def preprocess_ecg(signal):

    signal = signal.astype(
        np.float32
    )

    # Per-lead mean centering

    signal = (
        signal
        - np.mean(
            signal,
            axis=0,
            keepdims=True
        )
    )

    # Per-lead standardization

    std = np.std(
        signal,
        axis=0,
        keepdims=True
    )

    std[
        std == 0
    ] = 1.0

    signal = signal / std

    return signal.astype(
        np.float32
    )


# ============================================================
# 6. FIND EXACT VERIFIED RECORD
# ============================================================

print()
print("--- VERIFIED RECORD SEARCH ---")

verified_rows = df[
    df["filename_lr"].astype(str).str.endswith(
        VERIFIED_RECORD
    )
]

print(
    "Matching metadata records:",
    len(verified_rows)
)

if len(verified_rows) != 1:

    print(
        "Could not uniquely locate:",
        VERIFIED_RECORD
    )

    verified_record_status = "FAIL"

else:

    verified_row = verified_rows.iloc[0]

    print(
        "ECG ID:",
        verified_row["ecg_id"]
    )

    print(
        "Patient ID:",
        verified_row["patient_id"]
    )

    print(
        "Filename:",
        verified_row["filename_lr"]
    )

    print(
        "Fold:",
        verified_row["strat_fold"]
    )

    verified_record_status = "PASS"


# ============================================================
# 7. BUILD EXACT WAVEFORM PATH
# ============================================================

if verified_record_status == "PASS":

    relative_filename = Path(
        str(
            verified_row["filename_lr"]
        )
    )

    record_path = (
        BASE_DIR
        / relative_filename
    )

    print()
    print("--- WAVEFORM PATH ---")

    print(
        "Record path:",
        record_path
    )

    header_file = Path(
        str(record_path) + ".hea"
    )

    data_file = Path(
        str(record_path) + ".dat"
    )

    print(
        "Header exists:",
        header_file.exists()
    )

    print(
        "Data exists:",
        data_file.exists()
    )

else:

    header_file = None
    data_file = None
    record_path = None


# ============================================================
# 8. LOAD VERIFIED WAVEFORM
# ============================================================

waveform_status = "FAIL"

if (
    record_path is not None
    and header_file.exists()
    and data_file.exists()
):

    print()
    print("--- WAVEFORM LOADING ---")

    try:

        record = wfdb.rdrecord(
            str(record_path)
        )

        raw_signal = (
            record.p_signal
        )

        print(
            "Record name:",
            record.record_name
        )

        print(
            "Sampling frequency:",
            record.fs
        )

        print(
            "Raw waveform shape:",
            raw_signal.shape
        )

        processed_signal = (
            preprocess_ecg(
                raw_signal
            )
        )

        print(
            "Processed waveform shape:",
            processed_signal.shape
        )

        print(
            "Processed NaN values:",
            np.isnan(
                processed_signal
            ).sum()
        )

        print(
            "Processed infinite values:",
            np.isinf(
                processed_signal
            ).sum()
        )

        waveform_status = "PASS"

    except Exception as e:

        print(
            "Waveform loading error:",
            str(e)
        )

else:

    print()
    print(
        "Verified waveform files are not available."
    )


# ============================================================
# 9. TARGET VECTOR
# ============================================================

target_status = "FAIL"

if verified_record_status == "PASS":

    target = create_target_vector(
        verified_row["scp_codes"]
    )

    print()
    print("--- TARGET VECTOR ---")

    print(
        "Target order:",
        SUPERCLASSES
    )

    print(
        "Target vector:",
        target.astype(int).tolist()
    )

    if (
        target.shape == (5,)
        and np.all(
            np.isin(
                target,
                [0, 1]
            )
        )
    ):

        target_status = "PASS"

    print(
        "Target validation:",
        target_status
    )


# ============================================================
# 10. EXPERIMENTAL SPLIT
# ============================================================

print()
print("--- EXPERIMENTAL SPLIT ---")

train_df = df[
    df["strat_fold"].isin(
        TRAIN_FOLDS
    )
]

validation_df = df[
    df["strat_fold"]
    == VALIDATION_FOLD
]

test_df = df[
    df["strat_fold"]
    == TEST_FOLD
]

print(
    "Training folds:",
    TRAIN_FOLDS
)

print(
    "Validation fold:",
    VALIDATION_FOLD
)

print(
    "Test fold:",
    TEST_FOLD
)

print()
print(
    "Training ECG records:",
    len(train_df)
)

print(
    "Validation ECG records:",
    len(validation_df)
)

print(
    "Test ECG records:",
    len(test_df)
)

# ============================================================
# 11. PATIENT INTEGRITY
# ============================================================

train_patients = set(
    train_df["patient_id"]
)

validation_patients = set(
    validation_df["patient_id"]
)

test_patients = set(
    test_df["patient_id"]
)

train_validation_overlap = (
    train_patients
    & validation_patients
)

train_test_overlap = (
    train_patients
    & test_patients
)

validation_test_overlap = (
    validation_patients
    & test_patients
)

print()
print("--- PATIENT SPLIT INTEGRITY ---")

print(
    "Train/Validation overlap:",
    len(train_validation_overlap)
)

print(
    "Train/Test overlap:",
    len(train_test_overlap)
)

print(
    "Validation/Test overlap:",
    len(validation_test_overlap)
)

# ============================================================
# 12. VERIFY SPLIT OF VERIFIED RECORD
# ============================================================

print()
print("--- VERIFIED RECORD SPLIT ---")

if verified_record_status == "PASS":

    verified_fold = int(
        verified_row["strat_fold"]
    )

    print(
        "Verified record:",
        VERIFIED_RECORD
    )

    print(
        "Assigned fold:",
        verified_fold
    )

    if verified_fold in TRAIN_FOLDS:

        print(
            "Role: TRAINING"
        )

    elif verified_fold == VALIDATION_FOLD:

        print(
            "Role: VALIDATION"
        )

    elif verified_fold == TEST_FOLD:

        print(
            "Role: TEST"
        )

    else:

        print(
            "Role: UNKNOWN"
        )


# ============================================================
# 13. FINAL VALIDATION
# ============================================================

split_status = "PASS"

if len(
    train_validation_overlap
) != 0:

    split_status = "FAIL"

if len(
    train_test_overlap
) != 0:

    split_status = "FAIL"

if len(
    validation_test_overlap
) != 0:

    split_status = "FAIL"


if (
    verified_record_status == "PASS"
    and waveform_status == "PASS"
    and target_status == "PASS"
    and split_status == "PASS"
):

    overall_status = "PASS"

else:

    overall_status = "FAIL"


# ============================================================
# 14. FINAL RESULT
# ============================================================

print()
print("=" * 70)

print(
    "RECORD IDENTIFICATION:",
    verified_record_status
)

print(
    "WAVEFORM LOADER:",
    waveform_status
)

print(
    "TARGET VALIDATION:",
    target_status
)

print(
    "SPLIT VALIDATION:",
    split_status
)

print(
    "OVERALL RESULT:",
    overall_status
)

print("=" * 70)


# ============================================================
# 15. SAVE REPORT
# ============================================================

report = []

report.append(
    "EXPERIMENT 3C - PTB-XL DATASET LOADER AND SPLIT VALIDATION"
)

report.append("=" * 70)

report.append(
    f"Total ECG records: {len(df)}"
)

report.append(
    f"Unique patients: {df['patient_id'].nunique()}"
)

report.append("")

report.append(
    "VERIFIED RECORD"
)

report.append(
    f"Record: {VERIFIED_RECORD}"
)

if verified_record_status == "PASS":

    report.append(
        f"ECG ID: {verified_row['ecg_id']}"
    )

    report.append(
        f"Patient ID: {verified_row['patient_id']}"
    )

    report.append(
        f"Fold: {verified_row['strat_fold']}"
    )

    report.append(
        f"Filename: {verified_row['filename_lr']}"
    )

report.append("")

report.append(
    "EXPERIMENTAL SPLIT"
)

report.append(
    "Training folds: 1-8"
)

report.append(
    "Validation fold: 9"
)

report.append(
    "Test fold: 10"
)

report.append(
    f"Training ECGs: {len(train_df)}"
)

report.append(
    f"Validation ECGs: {len(validation_df)}"
)

report.append(
    f"Test ECGs: {len(test_df)}"
)

report.append("")

report.append(
    "PATIENT OVERLAP"
)

report.append(
    f"Train/Validation: {len(train_validation_overlap)}"
)

report.append(
    f"Train/Test: {len(train_test_overlap)}"
)

report.append(
    f"Validation/Test: {len(validation_test_overlap)}"
)

report.append("")

report.append(
    "TARGET"
)

if target_status == "PASS":

    report.append(
        f"Target vector: {target.astype(int).tolist()}"
    )

report.append("")

report.append(
    f"Record identification: {verified_record_status}"
)

report.append(
    f"Waveform loader: {waveform_status}"
)

report.append(
    f"Target validation: {target_status}"
)

report.append(
    f"Split validation: {split_status}"
)

report.append(
    f"Overall result: {overall_status}"
)

OUT_FILE.write_text(
    "\n".join(report),
    encoding="utf-8"
)

print()
print("Report saved to:")
print(OUT_FILE)

print()
print("Experiment 3C completed.")