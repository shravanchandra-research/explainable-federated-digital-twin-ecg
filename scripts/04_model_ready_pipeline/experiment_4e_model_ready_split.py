import os
import ast
import numpy as np
import pandas as pd

# ============================================================
# EXPERIMENT 4E
# MODEL-READY DATASET SPLIT AND INDEX VALIDATION
# ============================================================

BASE_DIR = os.environ.get("PTBXL_ROOT", r"C:\PTB-XL")

METADATA_FILE = os.path.join(
    BASE_DIR,
    "ptbxl_database.csv"
)

ALIGNMENT_FILE = os.path.join(
    BASE_DIR,
    "experiments",
    "4D_output",
    "experiment_4d_alignment.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "experiments",
    "4E_output"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

INDEX_FILE = os.path.join(
    OUTPUT_DIR,
    "experiment_4e_model_ready_index.csv"
)

SUMMARY_FILE = os.path.join(
    OUTPUT_DIR,
    "experiment_4e_split_summary.csv"
)

EXPECTED_SHAPE = (1000, 12)

TARGET_CLASSES = [
    "NORM",
    "MI",
    "STTC",
    "CD",
    "HYP"
]

TRAIN_FOLDS = list(range(1, 9))
VALIDATION_FOLD = 9
TEST_FOLD = 10

# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("EXPERIMENT 4E - MODEL-READY DATASET")
print("SPLIT AND INDEX VALIDATION")
print("=" * 70)

# ============================================================
# LOAD METADATA
# ============================================================

print("\nLoading PTB-XL metadata...")

df = pd.read_csv(METADATA_FILE)

print("Total metadata records:", len(df))

# ============================================================
# FOLD COLUMN
# ============================================================

if "strat_fold" in df.columns:
    FOLD_COLUMN = "strat_fold"
elif "fold" in df.columns:
    FOLD_COLUMN = "fold"
else:
    print("\nERROR: No fold column found.")
    print(list(df.columns))
    raise SystemExit(1)

print("Fold column:", FOLD_COLUMN)

# ============================================================
# REQUIRED METADATA COLUMNS
# ============================================================

required_columns = [
    "ecg_id",
    "patient_id",
    "filename_lr",
    "scp_codes",
    FOLD_COLUMN
]

missing_columns = [
    c for c in required_columns
    if c not in df.columns
]

if missing_columns:
    print("\nERROR: Missing required metadata columns:")
    print(missing_columns)
    raise SystemExit(1)

# ============================================================
# LOAD 4D ALIGNMENT REPORT
# ============================================================

print("\nLoading Experiment 4D alignment report...")

alignment = pd.read_csv(ALIGNMENT_FILE)

print("4D alignment records:", len(alignment))

# ============================================================
# REQUIRED 4D COLUMNS
# ============================================================

required_alignment_columns = [
    "ecg_id",
    "record",
    "patient_id",
    "fold",
    "NORM",
    "MI",
    "STTC",
    "CD",
    "HYP",
    "samples",
    "leads",
    "filename_lr",
    "processed_file"
]

missing_alignment_columns = [
    c for c in required_alignment_columns
    if c not in alignment.columns
]

if missing_alignment_columns:
    print("\nERROR: Missing 4D alignment columns:")
    print(missing_alignment_columns)
    raise SystemExit(1)

# ============================================================
# NORMALIZE ECG IDs
# ============================================================

alignment["ecg_id"] = pd.to_numeric(
    alignment["ecg_id"],
    errors="coerce"
)

alignment = alignment.dropna(
    subset=["ecg_id"]
).copy()

alignment["ecg_id"] = alignment["ecg_id"].astype(int)

# ============================================================
# CHECK DUPLICATE ALIGNMENT IDs
# ============================================================

duplicate_alignment = alignment[
    alignment["ecg_id"].duplicated(keep=False)
]

if len(duplicate_alignment) > 0:

    print("\nERROR: Duplicate ECG IDs found in 4D alignment.")

    print(
        duplicate_alignment["ecg_id"].tolist()
    )

    raise SystemExit(1)

# ============================================================
# SELECT EXACT RECORDS VERIFIED IN 4D
# ============================================================

validated_ids = set(
    alignment["ecg_id"].tolist()
)

print(
    "ECG IDs available from 4D:",
    len(validated_ids)
)

# ============================================================
# MERGE METADATA WITH 4D ALIGNMENT
# ============================================================

metadata_subset = df[
    df["ecg_id"].isin(validated_ids)
].copy()

metadata_subset = metadata_subset.sort_values(
    "ecg_id"
).reset_index(drop=True)

alignment = alignment.sort_values(
    "ecg_id"
).reset_index(drop=True)

print(
    "Records selected for 4E:",
    len(metadata_subset)
)

if len(metadata_subset) != len(alignment):

    print("\nERROR: Metadata and alignment record counts differ.")

    print(
        "Metadata records:",
        len(metadata_subset)
    )

    print(
        "Alignment records:",
        len(alignment)
    )

    raise SystemExit(1)

# ============================================================
# CREATE ALIGNMENT LOOKUP
# ============================================================

alignment_lookup = alignment.set_index(
    "ecg_id"
)

# ============================================================
# STORAGE
# ============================================================

records = []

failed_records = 0

missing_preprocessed = 0
npy_load_errors = 0
shape_errors = 0
finite_errors = 0
identity_errors = 0
patient_errors = 0
label_errors = 0
fold_errors = 0

# ============================================================
# VALIDATE EACH RECORD
# ============================================================

print("\nValidating model-ready records...")

for _, row in metadata_subset.iterrows():

    ecg_id = int(row["ecg_id"])

    # --------------------------------------------------------
    # GET 4D VERIFIED INFORMATION
    # --------------------------------------------------------

    if ecg_id not in alignment_lookup.index:

        failed_records += 1
        identity_errors += 1
        continue

    aligned = alignment_lookup.loc[ecg_id]

    # --------------------------------------------------------
    # PATIENT ID
    # --------------------------------------------------------

    patient_id = row["patient_id"]

    if pd.isna(patient_id):

        failed_records += 1
        patient_errors += 1
        continue

    patient_id = int(patient_id)

    # --------------------------------------------------------
    # FOLD
    # --------------------------------------------------------

    try:

        fold = int(row[FOLD_COLUMN])

    except Exception:

        failed_records += 1
        fold_errors += 1
        continue

    if fold < 1 or fold > 10:

        failed_records += 1
        fold_errors += 1
        continue

    # --------------------------------------------------------
    # VERIFY 4D FOLD
    # --------------------------------------------------------

    try:

        alignment_fold = int(aligned["fold"])

    except Exception:

        failed_records += 1
        fold_errors += 1
        continue

    if fold != alignment_fold:

        failed_records += 1
        fold_errors += 1
        continue

    # --------------------------------------------------------
    # PROCESSED FILE
    #
    # Use the exact path recorded by Experiment 4D.
    # This preserves the PTB-XL directory structure.
    # --------------------------------------------------------

    processed_file = str(
        aligned["processed_file"]
    )

    if not os.path.exists(processed_file):

        failed_records += 1
        missing_preprocessed += 1
        continue

    # --------------------------------------------------------
    # LOAD NPY
    # --------------------------------------------------------

    try:

        signal = np.load(
            processed_file
        )

    except Exception:

        failed_records += 1
        npy_load_errors += 1
        continue

    # --------------------------------------------------------
    # SHAPE
    # --------------------------------------------------------

    if signal.shape != EXPECTED_SHAPE:

        failed_records += 1
        shape_errors += 1
        continue

    # --------------------------------------------------------
    # FINITE VALUES
    # --------------------------------------------------------

    if not np.isfinite(signal).all():

        failed_records += 1
        finite_errors += 1
        continue

    # --------------------------------------------------------
    # RECORD IDENTITY
    # --------------------------------------------------------

    record_name = str(
        aligned["record"]
    )

    numeric_part = "".join(
        ch for ch in record_name
        if ch.isdigit()
    )

    try:

        record_ecg_id = int(
            numeric_part
        )

    except Exception:

        failed_records += 1
        identity_errors += 1
        continue

    if record_ecg_id != ecg_id:

        failed_records += 1
        identity_errors += 1
        continue

    # --------------------------------------------------------
    # PATIENT IDENTITY
    # --------------------------------------------------------

    aligned_patient = aligned["patient_id"]

    if pd.isna(aligned_patient):

        failed_records += 1
        patient_errors += 1
        continue

    if int(float(aligned_patient)) != patient_id:

        failed_records += 1
        patient_errors += 1
        continue

    # --------------------------------------------------------
    # LABEL VALIDATION
    # --------------------------------------------------------

    target_values = []

    label_valid = True

    for target in TARGET_CLASSES:

        try:

            value = int(
                aligned[target]
            )

        except Exception:

            label_valid = False
            break

        if value not in [0, 1]:

            label_valid = False
            break

        target_values.append(value)

    if not label_valid:

        failed_records += 1
        label_errors += 1
        continue

    # --------------------------------------------------------
    # SPLIT ASSIGNMENT
    # --------------------------------------------------------

    if fold in TRAIN_FOLDS:

        split = "train"

    elif fold == VALIDATION_FOLD:

        split = "validation"

    elif fold == TEST_FOLD:

        split = "test"

    else:

        failed_records += 1
        fold_errors += 1
        continue

    # --------------------------------------------------------
    # STORE MODEL-READY RECORD
    # --------------------------------------------------------

    records.append({

        "ecg_id": ecg_id,

        "patient_id": patient_id,

        "record_name": record_name,

        "preprocessed_file": processed_file,

        "fold": fold,

        "split": split,

        "NORM": target_values[0],

        "MI": target_values[1],

        "STTC": target_values[2],

        "CD": target_values[3],

        "HYP": target_values[4]

    })

# ============================================================
# CREATE RESULT DATAFRAME
# ============================================================

if len(records) == 0:

    print("\nERROR: No model-ready records were created.")

    print(
        "Missing preprocessed:",
        missing_preprocessed
    )

    print(
        "NPY load errors:",
        npy_load_errors
    )

    print(
        "Shape errors:",
        shape_errors
    )

    print(
        "Finite errors:",
        finite_errors
    )

    print(
        "Identity errors:",
        identity_errors
    )

    print(
        "Patient errors:",
        patient_errors
    )

    print(
        "Label errors:",
        label_errors
    )

    print(
        "Fold errors:",
        fold_errors
    )

    raise SystemExit(1)

result_df = pd.DataFrame(
    records
)

# ============================================================
# DUPLICATE CHECK
# ============================================================

duplicate_output = result_df[
    result_df["ecg_id"].duplicated(
        keep=False
    )
]

if len(duplicate_output) > 0:

    print(
        "\nERROR: Duplicate ECG IDs detected."
    )

    print(
        duplicate_output[
            "ecg_id"
        ].tolist()
    )

    raise SystemExit(1)

# ============================================================
# RECORD COUNT CHECK
# ============================================================

expected_records = len(validated_ids)

record_count_error = (
    len(result_df) != expected_records
)

# ============================================================
# SAVE MODEL-READY INDEX
# ============================================================

result_df = result_df.sort_values(
    "ecg_id"
).reset_index(drop=True)

result_df.to_csv(
    INDEX_FILE,
    index=False
)

# ============================================================
# SPLIT SUMMARY
# ============================================================

summary_records = []

for split_name in [
    "train",
    "validation",
    "test"
]:

    subset = result_df[
        result_df["split"] == split_name
    ]

    summary_records.append({

        "split": split_name,

        "records": len(subset),

        "NORM": int(
            subset["NORM"].sum()
        ),

        "MI": int(
            subset["MI"].sum()
        ),

        "STTC": int(
            subset["STTC"].sum()
        ),

        "CD": int(
            subset["CD"].sum()
        ),

        "HYP": int(
            subset["HYP"].sum()
        )

    })

summary_df = pd.DataFrame(
    summary_records
)

summary_df.to_csv(
    SUMMARY_FILE,
    index=False
)

# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("EXPERIMENT 4E SUMMARY")
print("=" * 70)

print(
    "Records selected        :",
    expected_records
)

print(
    "Successfully indexed    :",
    len(result_df)
)

print(
    "Failed records          :",
    failed_records
)

print(
    "Missing preprocessed    :",
    missing_preprocessed
)

print(
    "NPY load errors         :",
    npy_load_errors
)

print(
    "Shape errors            :",
    shape_errors
)

print(
    "Finite-value errors     :",
    finite_errors
)

print(
    "Record identity errors  :",
    identity_errors
)

print(
    "Patient ID errors       :",
    patient_errors
)

print(
    "Label errors            :",
    label_errors
)

print(
    "Fold errors             :",
    fold_errors
)

# ============================================================
# SPLIT DISTRIBUTION
# ============================================================

print("\nDataset split distribution:")

for split_name in [
    "train",
    "validation",
    "test"
]:

    count = len(
        result_df[
            result_df["split"]
            == split_name
        ]
    )

    print(
        f"  {split_name.capitalize():12s}: {count}"
    )

# ============================================================
# DIAGNOSTIC SUPERCLASS COUNTS
# ============================================================

print("\nDiagnostic superclass counts:")

for target in TARGET_CLASSES:

    print(
        f"  {target:5s}: "
        f"{int(result_df[target].sum())}"
    )

# ============================================================
# FOLD DISTRIBUTION
# ============================================================

print("\nFold distribution:")

fold_counts = (
    result_df["fold"]
    .value_counts()
    .sort_index()
)

for fold_number in range(1, 11):

    count = int(
        fold_counts.get(
            fold_number,
            0
        )
    )

    print(
        f"  Fold {fold_number:2d}: {count}"
    )

# ============================================================
# OUTPUT FILES
# ============================================================

print("\nModel-ready index:")
print(INDEX_FILE)

print("\nSplit summary:")
print(SUMMARY_FILE)

# ============================================================
# FINAL PASS CONDITION
# ============================================================

pass_condition = (

    len(result_df) == expected_records

    and failed_records == 0

    and missing_preprocessed == 0

    and npy_load_errors == 0

    and shape_errors == 0

    and finite_errors == 0

    and identity_errors == 0

    and patient_errors == 0

    and label_errors == 0

    and fold_errors == 0

    and not record_count_error

)

print("\n" + "=" * 70)

if pass_condition:

    print(
        "EXPERIMENT 4E RESULT : PASS"
    )

else:

    print(
        "EXPERIMENT 4E RESULT : REVIEW"
    )

print("=" * 70)