import os
import ast
import csv

import numpy as np
import pandas as pd


# ============================================================
# EXPERIMENT 4D
# PTB-XL PREPROCESSED DATASET INTEGRITY & LABEL ALIGNMENT
# ============================================================

ROOT = os.environ.get("PTBXL_ROOT", r"C:\PTB-XL")

DATABASE_FILE = os.path.join(
    ROOT,
    "ptbxl_database.csv"
)

SCP_FILE = os.path.join(
    ROOT,
    "scp_statements.csv"
)

PROCESSED_DIR = os.path.join(
    ROOT,
    "experiments",
    "4C_output",
    "preprocessed"
)

OUTPUT_DIR = os.path.join(
    ROOT,
    "experiments",
    "4D_output"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

N_RECORDS = 100

TARGET_CLASSES = [
    "NORM",
    "MI",
    "STTC",
    "CD",
    "HYP"
]


# ============================================================
# START
# ============================================================

print("=" * 70)
print("EXPERIMENT 4D - PREPROCESSED DATASET INTEGRITY")
print("AND LABEL ALIGNMENT")
print("=" * 70)


# ============================================================
# CHECK INPUT FILES
# ============================================================

print()
print("Checking input files...")

if not os.path.exists(DATABASE_FILE):

    print("ERROR: ptbxl_database.csv not found:")
    print(DATABASE_FILE)
    raise SystemExit(1)

if not os.path.exists(SCP_FILE):

    print("ERROR: scp_statements.csv not found:")
    print(SCP_FILE)
    raise SystemExit(1)

if not os.path.exists(PROCESSED_DIR):

    print("ERROR: Preprocessed directory not found:")
    print(PROCESSED_DIR)
    raise SystemExit(1)

print("Input files verified.")


# ============================================================
# LOAD PTB-XL DATABASE
# ============================================================

print()
print("Loading PTB-XL metadata...")

try:

    df = pd.read_csv(
        DATABASE_FILE
    )

except Exception as e:

    print("ERROR loading ptbxl_database.csv:")
    print(e)
    raise SystemExit(1)


print(
    "Total metadata records:",
    len(df)
)


# ============================================================
# DISPLAY COLUMNS
# ============================================================

print()
print("Metadata columns detected:")

for column in df.columns:

    print(
        "  -",
        column
    )


# ============================================================
# IDENTIFY FOLD COLUMN
# ============================================================

if "strat_fold" in df.columns:

    FOLD_COLUMN = "strat_fold"

elif "fold" in df.columns:

    FOLD_COLUMN = "fold"

else:

    print()
    print("ERROR: No fold column found.")
    print("Expected 'strat_fold' in PTB-XL metadata.")
    raise SystemExit(1)


print()
print(
    "Fold column used:",
    FOLD_COLUMN
)


# ============================================================
# VERIFY REQUIRED DATABASE COLUMNS
# ============================================================

required_columns = [
    "ecg_id",
    "patient_id",
    "filename_lr",
    "filename_hr",
    "scp_codes"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    print()
    print(
        "ERROR: Missing required columns:",
        missing_columns
    )

    raise SystemExit(1)


print()
print("All required metadata columns verified.")


# ============================================================
# LOAD SCP STATEMENTS
# ============================================================

print()
print("Loading SCP diagnostic statements...")

try:

    scp_df = pd.read_csv(
        SCP_FILE,
        index_col=0
    )

except Exception as e:

    print(
        "ERROR loading scp_statements.csv:"
    )

    print(e)

    raise SystemExit(1)


print(
    "SCP statements loaded:",
    len(scp_df)
)


# ============================================================
# VERIFY SCP CLASS COLUMN
# ============================================================

if "diagnostic_class" not in scp_df.columns:

    print()
    print(
        "ERROR: diagnostic_class column not found "
        "in scp_statements.csv"
    )

    print()
    print(
        "Available SCP columns:"
    )

    for column in scp_df.columns:

        print(
            "  -",
            column
        )

    raise SystemExit(1)


# ============================================================
# CREATE SCP CODE → DIAGNOSTIC SUPERCLASS MAP
# ============================================================

scp_to_class = {}

for scp_code, row in scp_df.iterrows():

    diagnostic_class = row[
        "diagnostic_class"
    ]

    if pd.notna(
        diagnostic_class
    ):

        diagnostic_class = str(
            diagnostic_class
        ).strip().upper()

        if diagnostic_class in TARGET_CLASSES:

            scp_to_class[
                str(scp_code).strip().upper()
            ] = diagnostic_class


print()
print(
    "SCP codes mapped to target classes:",
    len(scp_to_class)
)


# ============================================================
# SORT DATABASE BY ECG ID
# ============================================================

df = df.sort_values(
    by="ecg_id"
).reset_index(
    drop=True
)


# ============================================================
# SELECT FIRST 100 ECGs
# ============================================================

df_selected = df.head(
    N_RECORDS
).copy()


print()
print(
    "Records selected for validation:",
    len(df_selected)
)


# ============================================================
# COUNTERS
# ============================================================

successful = 0
failed = 0

missing_preprocessed = 0
load_errors = 0
shape_errors = 0
finite_errors = 0
record_identity_errors = 0
patient_errors = 0
fold_errors = 0
label_errors = 0


# ============================================================
# RESULT STORAGE
# ============================================================

results = []


# ============================================================
# PROCESS EACH ECG
# ============================================================

for index, row in df_selected.iterrows():

    ecg_id = int(
        row["ecg_id"]
    )

    print()
    print(
        "[{}/{}] ECG ID: {}".format(
            index + 1,
            len(df_selected),
            ecg_id
        )
    )


    # ========================================================
    # GET LOW-RESOLUTION FILENAME
    # ========================================================

    filename_lr = str(
        row["filename_lr"]
    ).replace(
        "\\",
        "/"
    )

    record_name = os.path.basename(
        filename_lr
    )


    # ========================================================
    # GET RECORD FOLDER
    # ========================================================

    path_parts = filename_lr.split(
        "/"
    )

    if len(path_parts) < 2:

        print(
            "  ERROR: Invalid filename_lr:",
            filename_lr
        )

        record_identity_errors += 1
        failed += 1

        continue


    folder_name = path_parts[-2]


    # ========================================================
    # PREPROCESSED NPY FILE
    # ========================================================

    processed_file = os.path.join(
        PROCESSED_DIR,
        folder_name,
        record_name + ".npy"
    )


    print(
        "  Record:",
        record_name
    )


    # ========================================================
    # CHECK NPY EXISTS
    # ========================================================

    if not os.path.exists(
        processed_file
    ):

        print(
            "  ERROR: Preprocessed file missing"
        )

        print(
            "  Expected:",
            processed_file
        )

        missing_preprocessed += 1
        failed += 1

        continue


    # ========================================================
    # LOAD NPY
    # ========================================================

    try:

        signal = np.load(
            processed_file
        )

    except Exception as e:

        print(
            "  ERROR loading NPY:",
            e
        )

        load_errors += 1
        failed += 1

        continue


    # ========================================================
    # VERIFY SHAPE
    # ========================================================

    if signal.shape != (
        1000,
        12
    ):

        print(
            "  ERROR: Invalid shape:",
            signal.shape
        )

        shape_errors += 1
        failed += 1

        continue


    # ========================================================
    # VERIFY FINITE VALUES
    # ========================================================

    if not np.all(
        np.isfinite(signal)
    ):

        print(
            "  ERROR: NaN or infinite values detected"
        )

        finite_errors += 1
        failed += 1

        continue


    # ========================================================
    # VERIFY RECORD IDENTITY
    # ========================================================

    try:

        record_numeric_id = int(
            record_name.split(
                "_"
            )[0]
        )

    except Exception:

        print(
            "  ERROR: Cannot extract numeric ID"
        )

        record_identity_errors += 1
        failed += 1

        continue


    if record_numeric_id != ecg_id:

        print(
            "  ERROR: ECG ID mismatch"
        )

        print(
            "  Metadata ECG ID:",
            ecg_id
        )

        print(
            "  Record numeric ID:",
            record_numeric_id
        )

        record_identity_errors += 1
        failed += 1

        continue


    # ========================================================
    # PATIENT ID
    # ========================================================

    patient_id = row[
        "patient_id"
    ]

    if pd.isna(
        patient_id
    ):

        print(
            "  ERROR: Missing patient ID"
        )

        patient_errors += 1
        failed += 1

        continue


    # ========================================================
    # STRATIFIED FOLD
    # ========================================================

    try:

        fold = int(
            row[FOLD_COLUMN]
        )

    except Exception:

        print(
            "  ERROR: Invalid fold value:",
            row[FOLD_COLUMN]
        )

        fold_errors += 1
        failed += 1

        continue


    if fold < 1 or fold > 10:

        print(
            "  ERROR: Fold outside 1-10:",
            fold
        )

        fold_errors += 1
        failed += 1

        continue


    # ========================================================
    # PARSE SCP CODES
    # ========================================================

    try:

        scp_codes = ast.literal_eval(
            str(
                row["scp_codes"]
            )
        )

        if not isinstance(
            scp_codes,
            dict
        ):

            raise ValueError(
                "scp_codes is not a dictionary"
            )

    except Exception as e:

        print(
            "  ERROR parsing scp_codes:",
            e
        )

        label_errors += 1
        failed += 1

        continue


    # ========================================================
    # CREATE FIVE-CLASS TARGET VECTOR
    #
    # Target order:
    #
    # [NORM, MI, STTC, CD, HYP]
    # ========================================================

    label_vector = {
        "NORM": 0,
        "MI": 0,
        "STTC": 0,
        "CD": 0,
        "HYP": 0
    }


    # ========================================================
    # MAP INDIVIDUAL SCP CODES TO SUPERCLASSES
    # ========================================================

    mapped_codes = []

    for scp_code in scp_codes.keys():

        code = str(
            scp_code
        ).strip().upper()

        if code in scp_to_class:

            superclass = scp_to_class[
                code
            ]

            label_vector[
                superclass
            ] = 1

            mapped_codes.append(
                code
            )


    # ========================================================
    # CREATE ORDERED VECTOR
    # ========================================================

    ordered_labels = [
        label_vector["NORM"],
        label_vector["MI"],
        label_vector["STTC"],
        label_vector["CD"],
        label_vector["HYP"]
    ]


    # ========================================================
    # CHECK LABEL VECTOR
    # ========================================================

    if len(
        ordered_labels
    ) != 5:

        print(
            "  ERROR: Invalid target vector"
        )

        label_errors += 1
        failed += 1

        continue


    # ========================================================
    # SUCCESS
    # ========================================================

    successful += 1


    results.append([
        ecg_id,
        record_name,
        patient_id,
        fold,

        label_vector["NORM"],
        label_vector["MI"],
        label_vector["STTC"],
        label_vector["CD"],
        label_vector["HYP"],

        signal.shape[0],
        signal.shape[1],

        filename_lr,
        processed_file,

        str(scp_codes),
        ",".join(mapped_codes)
    ])


    # ========================================================
    # DISPLAY
    # ========================================================

    print(
        "  Patient ID:",
        patient_id
    )

    print(
        "  Stratified fold:",
        fold
    )

    print(
        "  Original SCP codes:",
        len(scp_codes)
    )

    print(
        "  Mapped diagnostic codes:",
        len(mapped_codes)
    )

    print(
        "  Target [NORM, MI, STTC, CD, HYP]:",
        ordered_labels
    )

    print(
        "  Shape:",
        signal.shape
    )

    print(
        "  Status: PASS"
    )


# ============================================================
# SAVE ALIGNMENT REPORT
# ============================================================

alignment_file = os.path.join(
    OUTPUT_DIR,
    "experiment_4d_alignment.csv"
)


with open(
    alignment_file,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(
        f
    )

    writer.writerow([
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
        "processed_file",
        "scp_codes",
        "mapped_scp_codes"
    ])

    writer.writerows(
        results
    )


# ============================================================
# CALCULATE LABEL COUNTS
# ============================================================

label_counts = {
    "NORM": 0,
    "MI": 0,
    "STTC": 0,
    "CD": 0,
    "HYP": 0
}


for result in results:

    label_counts["NORM"] += int(
        result[4]
    )

    label_counts["MI"] += int(
        result[5]
    )

    label_counts["STTC"] += int(
        result[6]
    )

    label_counts["CD"] += int(
        result[7]
    )

    label_counts["HYP"] += int(
        result[8]
    )


# ============================================================
# FOLD DISTRIBUTION
# ============================================================

fold_counts = {}

for result in results:

    fold = int(
        result[3]
    )

    fold_counts[fold] = (
        fold_counts.get(
            fold,
            0
        ) + 1
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("EXPERIMENT 4D SUMMARY")
print("=" * 70)

print(
    "Records selected        :",
    len(df_selected)
)

print(
    "Successfully aligned    :",
    successful
)

print(
    "Failed records          :",
    failed
)

print(
    "Missing preprocessed    :",
    missing_preprocessed
)

print(
    "NPY load errors         :",
    load_errors
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
    record_identity_errors
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
# LABEL COUNTS
# ============================================================

print()
print(
    "Diagnostic superclass counts:"
)

print(
    "  NORM :",
    label_counts["NORM"]
)

print(
    "  MI   :",
    label_counts["MI"]
)

print(
    "  STTC :",
    label_counts["STTC"]
)

print(
    "  CD   :",
    label_counts["CD"]
)

print(
    "  HYP  :",
    label_counts["HYP"]
)


# ============================================================
# FOLD DISTRIBUTION
# ============================================================

print()
print(
    "Stratified fold distribution:"
)

for fold in sorted(
    fold_counts.keys()
):

    print(
        "  Fold {:2d}: {}".format(
            fold,