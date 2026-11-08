import os
import pandas as pd

# ============================================================
# EXPERIMENT 4F
# MODEL-READY DATASET CLASS DISTRIBUTION
# AND SPLIT INTEGRITY VALIDATION
# ============================================================

BASE_DIR = os.environ.get("PTBXL_ROOT", r"C:\PTB-XL")

INDEX_FILE = os.path.join(
    BASE_DIR,
    "experiments",
    "4E_output",
    "experiment_4e_model_ready_index.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "experiments",
    "4F_output"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

DISTRIBUTION_FILE = os.path.join(
    OUTPUT_DIR,
    "experiment_4f_class_distribution.csv"
)

OVERLAP_FILE = os.path.join(
    OUTPUT_DIR,
    "experiment_4f_overlap_report.csv"
)

TARGET_CLASSES = [
    "NORM",
    "MI",
    "STTC",
    "CD",
    "HYP"
]

EXPECTED_SPLITS = [
    "train",
    "validation",
    "test"
]

# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("EXPERIMENT 4F - CLASS DISTRIBUTION")
print("AND DATASET SPLIT INTEGRITY")
print("=" * 70)

# ============================================================
# LOAD 4E INDEX
# ============================================================

print("\nLoading Experiment 4E model-ready index...")

df = pd.read_csv(INDEX_FILE)

print("Records loaded:", len(df))

# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "ecg_id",
    "patient_id",
    "record_name",
    "preprocessed_file",
    "fold",
    "split"
] + TARGET_CLASSES

missing_columns = [
    c for c in required_columns
    if c not in df.columns
]

if missing_columns:
    print("\nERROR: Missing required columns:")
    print(missing_columns)
    raise SystemExit(1)

print("Required columns: PASS")

# ============================================================
# DUPLICATE ECG CHECK
# ============================================================

print("\nChecking duplicate ECG IDs...")

duplicate_ecg = df[
    df["ecg_id"].duplicated(keep=False)
]

ecg_duplicate_count = len(duplicate_ecg)

print(
    "Duplicate ECG records:",
    ecg_duplicate_count
)

# ============================================================
# PATIENT DISTRIBUTION
# ============================================================

print("\nChecking patient distribution...")

patient_counts = (
    df.groupby("patient_id")
    .size()
)

multiple_ecg_patients = patient_counts[
    patient_counts > 1
]

print(
    "Unique patients:",
    patient_counts.shape[0]
)

print(
    "Patients with multiple ECGs:",
    len(multiple_ecg_patients)
)

# ============================================================
# SPLIT VALIDATION
# ============================================================

print("\nValidating split values...")

invalid_split_values = sorted(
    set(df["split"].astype(str))
    - set(EXPECTED_SPLITS)
)

print(
    "Invalid split values:",
    len(invalid_split_values)
)

if invalid_split_values:
    print(invalid_split_values)

# ============================================================
# FOLD TO SPLIT VALIDATION
# ============================================================

print("\nValidating fold-to-split mapping...")

fold_split_errors = 0

for _, row in df.iterrows():

    fold = int(row["fold"])
    split = str(row["split"])

    if fold in range(1, 9):
        expected_split = "train"
    elif fold == 9:
        expected_split = "validation"
    elif fold == 10:
        expected_split = "test"
    else:
        expected_split = None

    if split != expected_split:
        fold_split_errors += 1

print(
    "Fold/split mapping errors:",
    fold_split_errors
)

# ============================================================
# ECG OVERLAP
# ============================================================

print("\nChecking ECG overlap between splits...")

train_ecg = set(
    df.loc[
        df["split"] == "train",
        "ecg_id"
    ]
)

validation_ecg = set(
    df.loc[
        df["split"] == "validation",
        "ecg_id"
    ]
)

test_ecg = set(
    df.loc[
        df["split"] == "test",
        "ecg_id"
    ]
)

train_validation_ecg_overlap = (
    train_ecg & validation_ecg
)

train_test_ecg_overlap = (
    train_ecg & test_ecg
)

validation_test_ecg_overlap = (
    validation_ecg & test_ecg
)

print(
    "Train/Validation ECG overlap:",
    len(train_validation_ecg_overlap)
)

print(
    "Train/Test ECG overlap:",
    len(train_test_ecg_overlap)
)

print(
    "Validation/Test ECG overlap:",
    len(validation_test_ecg_overlap)
)

# ============================================================
# PATIENT OVERLAP
# ============================================================

print("\nChecking patient overlap between splits...")

train_patients = set(
    df.loc[
        df["split"] == "train",
        "patient_id"
    ]
)

validation_patients = set(
    df.loc[
        df["split"] == "validation",
        "patient_id"
    ]
)

test_patients = set(
    df.loc[
        df["split"] == "test",
        "patient_id"
    ]
)

train_validation_patient_overlap = (
    train_patients & validation_patients
)

train_test_patient_overlap = (
    train_patients & test_patients
)

validation_test_patient_overlap = (
    validation_patients & test_patients
)

print(
    "Train/Validation patient overlap:",
    len(train_validation_patient_overlap)
)

print(
    "Train/Test patient overlap:",
    len(train_test_patient_overlap)
)

print(
    "Validation/Test patient overlap:",
    len(validation_test_patient_overlap)
)

# ============================================================
# LABEL VALIDATION
# ============================================================

print("\nValidating diagnostic labels...")

label_errors = 0

for target in TARGET_CLASSES:

    invalid_values = df[
        ~df[target].isin([0, 1])
    ]

    count = len(invalid_values)

    print(
        f"  {target:5s} invalid values: {count}"
    )

    label_errors += count

# ============================================================
# LABEL COUNT PER RECORD
# ============================================================

label_matrix = df[
    TARGET_CLASSES
].astype(int)

labels_per_record = (
    label_matrix.sum(axis=1)
)

zero_label_records = int(
    (labels_per_record == 0).sum()
)

single_label_records = int(
    (labels_per_record == 1).sum()
)

multi_label_records = int(
    (labels_per_record >= 2).sum()
)

print(
    "\nRecords with zero diagnostic labels:",
    zero_label_records
)

print(
    "Records with one diagnostic label:",
    single_label_records
)

print(
    "Records with multiple labels:",
    multi_label_records
)

# ============================================================
# CLASS DISTRIBUTION BY SPLIT
# ============================================================

print("\nDiagnostic class distribution by split:")

distribution_records = []

for split_name in EXPECTED_SPLITS:

    subset = df[
        df["split"] == split_name
    ]

    record_count = len(subset)

    row = {
        "split": split_name,
        "records": record_count
    }

    print("\n" + split_name.upper())

    print(
        "  Records:",
        record_count
    )

    for target in TARGET_CLASSES:

        positive_count = int(
            subset[target].sum()
        )

        if record_count > 0:
            percentage = (
                positive_count /
                record_count * 100
            )
        else:
            percentage = 0.0

        row[target] = positive_count

        row[target + "_percent"] = round(
            percentage,
            2
        )

        print(
            f"  {target:5s}: "
            f"{positive_count:3d} "
            f"({percentage:.2f}%)"
        )

    distribution_records.append(row)

distribution_df = pd.DataFrame(
    distribution_records
)

distribution_df.to_csv(
    DISTRIBUTION_FILE,
    index=False
)

# ============================================================
# OVERLAP REPORT
# ============================================================

overlap_records = [
    {
        "comparison": "train_validation_ecg",
        "overlap_count": len(
            train_validation_ecg_overlap
        )
    },
    {
        "comparison": "train_test_ecg",
        "overlap_count": len(
            train_test_ecg_overlap
        )
    },
    {
        "comparison": "validation_test_ecg",
        "overlap_count": len(
            validation_test_ecg_overlap
        )
    },
    {
        "comparison": "train_validation_patient",
        "overlap_count": len(
            train_validation_patient_overlap
        )
    },
    {
        "comparison": "train_test_patient",
        "overlap_count": len(
            train_test_patient_overlap
        )
    },
    {
        "comparison": "validation_test_patient",
        "overlap_count": len(
            validation_test_patient_overlap
        )
    }
]

overlap_df = pd.DataFrame(
    overlap_records
)

overlap_df.to_csv(
    OVERLAP_FILE,
    index=False
)

# ============================================================
# SPLIT COUNTS
# ============================================================

train_count = len(
    df[df["split"] == "train"]
)

validation_count = len(
    df[df["split"] == "validation"]
)

test_count = len(
    df[df["split"] == "test"]
)

total_split_count = (
    train_count
    + validation_count
    + test_count
)

total_count_error = (
    total_split_count != len(df)
)

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("EXPERIMENT 4F SUMMARY")
print("=" * 70)

print(
    "Total records             :",
    len(df)
)

print(
    "Training records          :",
    train_count
)

print(
    "Validation records        :",
    validation_count
)

print(
    "Testing records           :",
    test_count
)

print(
    "Unique patients           :",
    patient_counts.shape[0]
)

print(
    "Patients with multiple ECGs:",
    len(multiple_ecg_patients)
)

print(
    "Duplicate ECG IDs         :",
    ecg_duplicate_count
)

print(
    "Fold/split errors         :",
    fold_split_errors
)

print(
    "Invalid split values      :",
    len(invalid_split_values)
)

print(
    "Label errors              :",
    label_errors
)

print(
    "Zero-label records        :",
    zero_label_records
)

print(
    "Single-label records      :",
    single_label_records
)

print(
    "Multi-label records       :",
    multi_label_records
)

print("\nECG overlap:")

print(
    "  Train/Validation:",
    len(train_validation_ecg_overlap)
)

print(
    "  Train/Test:",
    len(train_test_ecg_overlap)
)

print(
    "  Validation/Test:",
    len(validation_test_ecg_overlap)
)

print("\nPatient overlap:")

print(
    "  Train/Validation:",
    len(train_validation_patient_overlap)
)

print(
    "  Train/Test:",
    len(train_test_patient_overlap)
)

print(
    "  Validation/Test:",
    len(validation_test_patient_overlap)
)

print("\nClass distribution file:")
print(DISTRIBUTION_FILE)

print("\nOverlap report:")
print(OVERLAP_FILE)

# ============================================================
# FINAL PASS CONDITION
# ============================================================

pass_condition = (
    len(df) == 100
    and total_split_count == len(df)
    and ecg_duplicate_count == 0
    and fold_split_errors == 0
    and len(invalid_split_values) == 0
    and label_errors == 0
    and len(train_validation_ecg_overlap) == 0
    and len(train_test_ecg_overlap) == 0
    and len(validation_test_ecg_overlap) == 0
    and len(train_validation_patient_overlap) == 0
    and len(train_test_patient_overlap) == 0
    and len(validation_test_patient_overlap) == 0
)

print("\n" + "=" * 70)

if pass_condition:

    print(
        "EXPERIMENT 4F RESULT : PASS"
    )

else:

    print(
        "EXPERIMENT 4F RESULT : REVIEW"
    )

print("=" * 70)