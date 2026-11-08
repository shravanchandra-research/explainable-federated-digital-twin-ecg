import os
import pandas as pd
import ast
from pathlib import Path

# ============================================================
# EXPERIMENT 3B
# PTB-XL MODEL-READY MULTI-LABEL TARGET CONSTRUCTION
# CORRECTED VERSION
# ============================================================

BASE_DIR = Path(os.environ.get("PTBXL_ROOT", r"C:\PTB-XL"))

DB_FILE = BASE_DIR / "ptbxl_database.csv"
SCP_FILE = BASE_DIR / "scp_statements.csv"

OUT_FILE = (
    BASE_DIR
    / "experiments"
    / "experiment_3b_label_construction.txt"
)

SUPERCLASSES = [
    "NORM",
    "MI",
    "STTC",
    "CD",
    "HYP"
]

print("=" * 70)
print("EXPERIMENT 3B - MODEL-READY MULTI-LABEL TARGET CONSTRUCTION")
print("=" * 70)

# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv(DB_FILE)
scp = pd.read_csv(SCP_FILE)

print()
print("--- DATASET ---")

print("Total ECG records :", len(df))
print("Unique patients   :", df["patient_id"].nunique())

# ============================================================
# 2. IDENTIFY DIAGNOSTIC SCP CODES
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

print()
print("--- SCP MAPPING ---")

print(
    "Diagnostic SCP codes:",
    len(diagnostic_scp)
)

# ============================================================
# 3. BUILD SCP CODE -> SUPERCLASS MAPPING
# ============================================================

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

            code_to_class[code] = diagnostic_class

print(
    "Mapped diagnostic codes:",
    len(code_to_class)
)

# ============================================================
# 4. PARSE SCP_CODES
# ============================================================

def parse_scp_codes(value):

    if pd.isna(value):
        return {}

    try:

        parsed = ast.literal_eval(
            str(value)
        )

        if isinstance(parsed, dict):

            return parsed

        return {}

    except Exception:

        return {}


# ============================================================
# 5. CONSTRUCT MULTI-LABEL TARGETS
# ============================================================

records = []

unknown_codes = set()

parse_failures = 0

for _, row in df.iterrows():

    raw_codes = row["scp_codes"]

    codes = parse_scp_codes(
        raw_codes
    )

    if raw_codes is not None:

        if pd.notna(raw_codes):

            if not isinstance(codes, dict):

                parse_failures += 1

    classes = set()

    for code in codes.keys():

        code = str(code).strip()

        if code in code_to_class:

            diagnostic_class = (
                code_to_class[code]
            )

            classes.add(
                diagnostic_class
            )

        else:

            unknown_codes.add(code)

    ordered_classes = [
        label
        for label in SUPERCLASSES
        if label in classes
    ]

    label_vector = [
        1 if label in classes else 0
        for label in SUPERCLASSES
    ]

    if len(ordered_classes) == 0:

        combination = "NONE"

    else:

        combination = "+".join(
            ordered_classes
        )

    records.append(
        {
            "ecg_id": row["ecg_id"],
            "patient_id": row["patient_id"],
            "strat_fold": row["strat_fold"],
            "diagnostic_classes": combination,
            "NORM": label_vector[0],
            "MI": label_vector[1],
            "STTC": label_vector[2],
            "CD": label_vector[3],
            "HYP": label_vector[4]
        }
    )

labels_df = pd.DataFrame(
    records
)

# ============================================================
# 6. LABEL MATRIX
# ============================================================

print()
print("--- LABEL MATRIX ---")

print(
    "Number of ECGs:",
    len(labels_df)
)

print(
    "Number of label columns:",
    len(SUPERCLASSES)
)

print(
    "Label matrix shape:",
    labels_df[
        SUPERCLASSES
    ].shape
)

# ============================================================
# 7. CLASS COUNTS
# ============================================================

print()
print("--- CLASS COUNTS ---")

class_counts = {}

for label in SUPERCLASSES:

    count = int(
        labels_df[label].sum()
    )

    class_counts[label] = count

    percentage = (
        count
        / len(labels_df)
        * 100
    )

    print(
        f"{label:5s}: "
        f"{count:5d} "
        f"({percentage:.2f}%)"
    )

# ============================================================
# 8. LABEL CARDINALITY
# ============================================================

labels_df["label_count"] = (
    labels_df[
        SUPERCLASSES
    ].sum(axis=1)
)

print()
print("--- LABEL CARDINALITY ---")

cardinality_counts = {}

for number in range(0, 6):

    count = int(
        (
            labels_df["label_count"]
            == number
        ).sum()
    )

    cardinality_counts[
        number
    ] = count

    if count > 0:

        percentage = (
            count
            / len(labels_df)
            * 100
        )

        print(
            f"{number} labels: "
            f"{count:5d} "
            f"({percentage:.2f}%)"
        )

# ============================================================
# 9. TOP COMBINATIONS
# ============================================================

print()
print("--- TOP LABEL COMBINATIONS ---")

combination_counts = (
    labels_df[
        "diagnostic_classes"
    ]
    .value_counts()
)

print(
    combination_counts
    .head(20)
    .to_string()
)

# ============================================================
# 10. NONE RECORDS
# ============================================================

none_count = int(
    (
        labels_df[
            "diagnostic_classes"
        ]
        == "NONE"
    ).sum()
)

print()
print("--- NONE LABEL CHECK ---")

print(
    "Records with no diagnostic superclass:",
    none_count
)

print(
    "NONE records are represented as:",
    "[0, 0, 0, 0, 0]"
)

# ============================================================
# 11. DUPLICATE ECG CHECK
# ============================================================

duplicate_ecg_ids = int(
    labels_df[
        "ecg_id"
    ].duplicated().sum()
)

print()
print("--- DUPLICATE CHECK ---")

print(
    "Duplicate ECG IDs:",
    duplicate_ecg_ids
)

# ============================================================
# 12. PATIENT/FOLD CHECK
# ============================================================

patient_fold_counts = (
    labels_df
    .groupby("patient_id")
    ["strat_fold"]
    .nunique()
)

cross_fold_patients = int(
    (
        patient_fold_counts > 1
    ).sum()
)

print()
print("--- PATIENT/FOLD CHECK ---")

print(
    "Patients appearing in multiple folds:",
    cross_fold_patients
)

# ============================================================
# 13. LABEL VALUE CHECK
# ============================================================

invalid_values = []

for label in SUPERCLASSES:

    values = set(
        labels_df[label].unique()
    )

    if not values.issubset({0, 1}):

        invalid_values.append(
            (label, values)
        )

print()
print("--- LABEL VALUE CHECK ---")

if len(invalid_values) == 0:

    print(
        "All label columns contain only 0 and 1."
    )

    label_value_status = "PASS"

else:

    print(
        "Invalid values detected:"
    )

    for item in invalid_values:

        print(item)

    label_value_status = "FAIL"

# ============================================================
# 14. PARSING CHECK
# ============================================================

print()
print("--- SCP PARSING CHECK ---")

print(
    "Parse failures:",
    parse_failures
)

if parse_failures == 0:

    parsing_status = "PASS"

else:

    parsing_status = "FAIL"

print(
    "Parsing status:",
    parsing_status
)

# ============================================================
# 15. CONSISTENCY CHECK WITH EXPERIMENT 2B
# ============================================================

# Experiment 2B established these counts.
# They are used here only as an internal consistency check.

expected_counts = {
    "NORM": 9514,
    "MI": 5469,
    "STTC": 5235,
    "CD": 4898,
    "HYP": 2649
}

print()
print("--- EXPERIMENT 2B CONSISTENCY CHECK ---")

consistency_status = "PASS"

for label in SUPERCLASSES:

    actual = class_counts[label]

    expected = expected_counts[label]

    difference = actual - expected

    print(
        f"{label:5s} "
        f"Expected={expected:5d} "
        f"Actual={actual:5d} "
        f"Difference={difference:5d}"
    )

    if actual != expected:

        consistency_status = "FAIL"

print(
    "Consistency status:",
    consistency_status
)

# ============================================================
# 16. OVERALL VALIDATION
# ============================================================

if (
    len(labels_df) == len(df)
    and duplicate_ecg_ids == 0
    and cross_fold_patients == 0
    and label_value_status == "PASS"
    and parsing_status == "PASS"
    and consistency_status == "PASS"
):

    overall_status = "PASS"

else:

    overall_status = "FAIL"

print()
print("=" * 70)

print(
    "OVERALL RESULT:",
    overall_status
)

print("=" * 70)

# ============================================================
# 17. SAVE REPORT
# ============================================================

report = []

report.append(
    "EXPERIMENT 3B - MODEL-READY MULTI-LABEL TARGET CONSTRUCTION"
)

report.append("=" * 70)

report.append(
    f"Total ECG records: {len(df)}"
)

report.append(
    f"Unique patients: {df['patient_id'].nunique()}"
)

report.append(
    f"Diagnostic SCP codes: {len(diagnostic_scp)}"
)

report.append(
    f"Mapped diagnostic codes: {len(code_to_class)}"
)

report.append("")

report.append(
    "MODEL TARGET ORDER:"
)

report.append(
    "[NORM, MI, STTC, CD, HYP]"
)

report.append("")

report.append(
    "CLASS COUNTS:"
)

for label in SUPERCLASSES:

    count = class_counts[label]

    percentage = (
        count
        / len(labels_df)
        * 100
    )

    report.append(
        f"{label}: "
        f"{count} "
        f"({percentage:.2f}%)"
    )

report.append("")

report.append(
    f"NONE records: {none_count}"
)

report.append(
    f"Duplicate ECG IDs: {duplicate_ecg_ids}"
)

report.append(
    f"Cross-fold patients: {cross_fold_patients}"
)

report.append(
    f"Parse failures: {parse_failures}"
)

report.append(
    f"Label validation: {label_value_status}"
)

report.append(
    f"Parsing validation: {parsing_status}"
)

report.append(
    f"Experiment 2B consistency: {consistency_status}"
)

report.append("")

report.append(
    "TARGET EXAMPLES:"
)

report.append(
    "NORM -> [1, 0, 0, 0, 0]"
)

report.append(
    "MI -> [0, 1, 0, 0, 0]"
)

report.append(
    "CD + MI + STTC -> [0, 1, 1, 1, 0]"
)

report.append(
    "NONE -> [0, 0, 0, 0, 0]"
)

report.append("")

report.append(
    f"OVERALL RESULT: {overall_status}"
)

OUT_FILE.write_text(
    "\n".join(report),
    encoding="utf-8"
)

print()
print("Report saved to:")
print(OUT_FILE)

print()
print("Experiment 3B completed.")