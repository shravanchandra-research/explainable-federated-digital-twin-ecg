import os
import numpy as np
import pandas as pd

# ============================================================
# EXPERIMENT 4G
# XAI EXPLANATION STABILITY - INPUT VALIDATION
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
    "4G_output"
)

INPUT_OUTPUT_DIR = os.path.join(
    OUTPUT_DIR,
    "xai_input_validation"
)

os.makedirs(INPUT_OUTPUT_DIR, exist_ok=True)

REPORT_FILE = os.path.join(
    OUTPUT_DIR,
    "experiment_4g_xai_input_validation.csv"
)

TARGET_CLASSES = [
    "NORM",
    "MI",
    "STTC",
    "CD",
    "HYP"
]

print("=" * 70)
print("EXPERIMENT 4G")
print("XAI EXPLANATION STABILITY - INPUT VALIDATION")
print("=" * 70)

# ------------------------------------------------------------
# LOAD MODEL-READY INDEX
# ------------------------------------------------------------

print("\nLoading Experiment 4E model-ready index...")

df = pd.read_csv(INDEX_FILE)

print("Records loaded:", len(df))

required_columns = [
    "ecg_id",
    "patient_id",
    "record_name",
    "preprocessed_file",
    "fold",
    "split"
] + TARGET_CLASSES

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    print("\nERROR: Missing required columns:")
    print(missing_columns)
    raise SystemExit(1)

print("Required columns: PASS")

# ------------------------------------------------------------
# SELECT RECORDS
# ------------------------------------------------------------

# Use the complete 100-record model-ready sample
selected_df = df.copy()

print(
    "\nRecords selected for XAI input validation:",
    len(selected_df)
)

# ------------------------------------------------------------
# VALIDATION COUNTERS
# ------------------------------------------------------------

successful = 0
missing_files = 0
load_errors = 0
shape_errors = 0
finite_errors = 0
sample_errors = 0
lead_errors = 0
label_errors = 0

report_records = []

# ------------------------------------------------------------
# PROCESS EACH ECG
# ------------------------------------------------------------

for _, row in selected_df.iterrows():

    ecg_id = int(row["ecg_id"])
    record_name = str(row["record_name"])
    patient_id = row["patient_id"]
    split = str(row["split"])

    processed_file = str(
        row["preprocessed_file"]
    )

    status = "PASS"
    error_reason = ""

    # --------------------------------------------------------
    # FILE EXISTENCE
    # --------------------------------------------------------

    if not os.path.exists(processed_file):

        missing_files += 1

        status = "FAIL"
        error_reason = "Preprocessed file missing"

        report_records.append({
            "ecg_id": ecg_id,
            "record_name": record_name,
            "patient_id": patient_id,
            "split": split,
            "processed_file": processed_file,
            "samples": 0,
            "leads": 0,
            "finite": False,
            "status": status,
            "error": error_reason
        })

        continue

    # --------------------------------------------------------
    # LOAD NPY
    # --------------------------------------------------------

    try:

        signal = np.load(
            processed_file
        )

    except Exception as e:

        load_errors += 1

        status = "FAIL"
        error_reason = (
            "NPY load error: " + str(e)
        )

        report_records.append({
            "ecg_id": ecg_id,
            "record_name": record_name,
            "patient_id": patient_id,
            "split": split,
            "processed_file": processed_file,
            "samples": 0,
            "leads": 0,
            "finite": False,
            "status": status,
            "error": error_reason
        })

        continue

    # --------------------------------------------------------
    # SHAPE VALIDATION
    # --------------------------------------------------------

    if signal.shape != (1000, 12):

        shape_errors += 1

        status = "FAIL"
        error_reason = (
            "Invalid shape: "
            + str(signal.shape)
        )

    # --------------------------------------------------------
    # FINITE VALUE VALIDATION
    # --------------------------------------------------------

    finite = bool(
        np.isfinite(signal).all()
    )

    if not finite:

        finite_errors += 1

        status = "FAIL"

        if error_reason:
            error_reason += "; "

        error_reason += (
            "NaN or infinite values"
        )

    # --------------------------------------------------------
    # SAMPLE VALIDATION
    # --------------------------------------------------------

    if signal.shape[0] != 1000:

        sample_errors += 1

        status = "FAIL"

        if error_reason:
            error_reason += "; "

        error_reason += (
            "Sample count != 1000"
        )

    # --------------------------------------------------------
    # LEAD VALIDATION
    # --------------------------------------------------------

    if signal.shape[1] != 12:

        lead_errors += 1

        status = "FAIL"

        if error_reason:
            error_reason += "; "

        error_reason += (
            "Lead count != 12"
        )

    # --------------------------------------------------------
    # LABEL VALIDATION
    # --------------------------------------------------------

    labels_valid = True

    for target in TARGET_CLASSES:

        value = row[target]

        if value not in [0, 1]:

            labels_valid = False

            break

    if not labels_valid:

        label_errors += 1

        status = "FAIL"

        if error_reason:
            error_reason += "; "

        error_reason += (
            "Invalid diagnostic label"
        )

    # --------------------------------------------------------
    # SAVE VALID INPUT
    # --------------------------------------------------------

    if status == "PASS":

        successful += 1

        # Create a copy for XAI processing.
        # Original preprocessed ECG is never modified.
        xai_file = os.path.join(
            INPUT_OUTPUT_DIR,
            record_name + ".npy"
        )

        np.save(
            xai_file,
            signal.astype(
                np.float32
            )
        )

    report_records.append({
        "ecg_id": ecg_id,
        "record_name": record_name,
        "patient_id": patient_id,
        "split": split,
        "processed_file": processed_file,
        "samples": signal.shape[0],
        "leads": signal.shape[1],
        "finite": finite,
        "status": status,
        "error": error_reason
    })

# ------------------------------------------------------------
# SAVE REPORT
# ------------------------------------------------------------

report_df = pd.DataFrame(
    report_records
)

report_df.to_csv(
    REPORT_FILE,
    index=False
)

# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("EXPERIMENT 4G SUMMARY")
print("=" * 70)

print(
    "Records selected       :",
    len(selected_df)
)

print(
    "Successfully validated :",
    successful
)

print(
    "Missing files          :",
    missing_files
)

print(
    "NPY load errors        :",
    load_errors
)

print(
    "Shape errors           :",
    shape_errors
)

print(
    "Finite-value errors    :",
    finite_errors
)

print(
    "Sample-count errors    :",
    sample_errors
)

print(
    "Lead-count errors      :",
    lead_errors
)

print(
    "Label errors           :",
    label_errors
)

print("\nXAI input directory:")
print(INPUT_OUTPUT_DIR)

print("\nValidation report:")
print(REPORT_FILE)

# ------------------------------------------------------------
# PASS CONDITION
# ------------------------------------------------------------

pass_condition = (
    len(selected_df) == 100
    and successful == 100
    and missing_files == 0
    and load_errors == 0
    and shape_errors == 0
    and finite_errors == 0
    and sample_errors == 0
    and lead_errors == 0
    and label_errors == 0
)

print("\n" + "=" * 70)

if pass_condition:

    print(
        "EXPERIMENT 4G INPUT VALIDATION : PASS"
    )

else:

    print(
        "EXPERIMENT 4G INPUT VALIDATION : REVIEW"
    )

print("=" * 70)