import os
import pandas as pd
from pathlib import Path

# ============================================================
# EXPERIMENT 2C
# PTB-XL PATIENT-LEVEL AND FOLD INTEGRITY ANALYSIS
# ============================================================

BASE_DIR = Path(os.environ.get("PTBXL_ROOT", r"C:\PTB-XL"))
DB_FILE = BASE_DIR / "ptbxl_database.csv"
OUT_FILE = BASE_DIR / "experiments" / "experiment_2c_patient_fold_analysis.txt"

print("=" * 70)
print("EXPERIMENT 2C - PTB-XL PATIENT/FOLD INTEGRITY ANALYSIS")
print("=" * 70)

# ------------------------------------------------------------
# 1. LOAD DATABASE
# ------------------------------------------------------------

df = pd.read_csv(DB_FILE)

print()
print("Total ECG records :", len(df))
print("Unique patients   :", df["patient_id"].nunique())

# ------------------------------------------------------------
# 2. ECGs PER PATIENT
# ------------------------------------------------------------

patient_counts = df.groupby("patient_id").size()

print()
print("--- PATIENT ECG DISTRIBUTION ---")
print("Patients with 1 ECG :", (patient_counts == 1).sum())
print("Patients with 2 ECGs:", (patient_counts == 2).sum())
print("Patients with 3 ECGs:", (patient_counts == 3).sum())
print("Patients with 4 ECGs:", (patient_counts == 4).sum())
print("Patients with 5+ ECGs:", (patient_counts >= 5).sum())

print("Maximum ECGs for one patient:", patient_counts.max())

# ------------------------------------------------------------
# 3. PATIENTS WITH MULTIPLE ECGs
# ------------------------------------------------------------

multi_patients = patient_counts[patient_counts > 1]

print()
print("--- MULTIPLE-ECG PATIENTS ---")
print("Patients with multiple ECGs:", len(multi_patients))
print(
    "Percentage of patients with multiple ECGs:",
    round(len(multi_patients) / df["patient_id"].nunique() * 100, 2),
    "%"
)

# ------------------------------------------------------------
# 4. PATIENTS SPANNING MULTIPLE FOLDS
# ------------------------------------------------------------

patient_fold_counts = df.groupby("patient_id")["strat_fold"].nunique()

cross_fold_patients = patient_fold_counts[patient_fold_counts > 1]

print()
print("--- PATIENT/FOLD DISTRIBUTION ---")
print(
    "Patients appearing in multiple folds:",
    len(cross_fold_patients)
)

print(
    "Percentage of patients appearing in multiple folds:",
    round(
        len(cross_fold_patients)
        / df["patient_id"].nunique()
        * 100,
        2
    ),
    "%"
)

# ------------------------------------------------------------
# 5. FOLD PAIRS
# ------------------------------------------------------------

print()
print("--- EXAMPLES OF PATIENTS APPEARING IN MULTIPLE FOLDS ---")

example_patients = cross_fold_patients.head(10).index

for patient in example_patients:

    patient_rows = df[df["patient_id"] == patient]

    print()
    print("Patient:", patient)

    print(
        patient_rows[
            [
                "ecg_id",
                "patient_id",
                "strat_fold",
                "filename_lr",
                "filename_hr"
            ]
        ].to_string(index=False)
    )

# ------------------------------------------------------------
# 6. FOLD RECORD COUNTS
# ------------------------------------------------------------

fold_counts = df["strat_fold"].value_counts().sort_index()

print()
print("--- ECG COUNT BY FOLD ---")

for fold, count in fold_counts.items():
    print(f"Fold {fold:2d}: {count:5d}")

# ------------------------------------------------------------
# 7. PATIENT COUNTS BY FOLD
# ------------------------------------------------------------

patient_fold_table = (
    df.groupby("strat_fold")["patient_id"]
    .nunique()
    .sort_index()
)

print()
print("--- UNIQUE PATIENT COUNT BY FOLD ---")

for fold, count in patient_fold_table.items():
    print(f"Fold {fold:2d}: {count:5d}")

# ------------------------------------------------------------
# 8. CROSS-FOLD PATIENT DETAILS
# ------------------------------------------------------------

cross_fold_details = (
    df.groupby("patient_id")["strat_fold"]
    .agg(["count", "nunique"])
)

cross_fold_details = cross_fold_details[
    cross_fold_details["nunique"] > 1
]

print()
print("--- CROSS-FOLD PATIENT SUMMARY ---")
print(
    "Number of cross-fold patients:",
    len(cross_fold_details)
)

if len(cross_fold_details) > 0:

    print()
    print("Distribution of number of folds per patient:")

    print(
        cross_fold_details["nunique"]
        .value_counts()
        .sort_index()
        .to_string()
    )

# ------------------------------------------------------------
# 9. RECOMMENDED SPLIT STRATEGY
# ------------------------------------------------------------

print()
print("--- DATA SPLIT RECOMMENDATION ---")

print(
    "For patient-independent evaluation, patient IDs should be kept"
)
print(
    "within a single partition whenever possible."
)

print(
    "Random ECG-level splitting should be avoided when the same"
)
print(
    "patient can contribute multiple ECG records."
)

print(
    "PTB-XL stratified folds should be inspected before selecting"
)
print(
    "training, validation, and test partitions."
)

# ------------------------------------------------------------
# 10. SAVE REPORT
# ------------------------------------------------------------

report_lines = []

report_lines.append(
    "EXPERIMENT 2C - PTB-XL PATIENT/FOLD INTEGRITY ANALYSIS"
)

report_lines.append("=" * 70)

report_lines.append(
    f"Total ECG records: {len(df)}"
)

report_lines.append(
    f"Unique patients: {df['patient_id'].nunique()}"
)

report_lines.append(
    f"Patients with multiple ECGs: {len(multi_patients)}"
)

report_lines.append(
    f"Patients appearing in multiple folds: {len(cross_fold_patients)}"
)

report_lines.append("")

report_lines.append("ECG COUNT BY FOLD")

for fold, count in fold_counts.items():

    report_lines.append(
        f"Fold {fold}: {count}"
    )

report_lines.append("")

report_lines.append("UNIQUE PATIENT COUNT BY FOLD")

for fold, count in patient_fold_table.items():

    report_lines.append(
        f"Fold {fold}: {count}"
    )

report_lines.append("")

report_lines.append("PATIENT ECG DISTRIBUTION")

report_lines.append(
    f"1 ECG: {(patient_counts == 1).sum()}"
)

report_lines.append(
    f"2 ECGs: {(patient_counts == 2).sum()}"
)

report_lines.append(
    f"3 ECGs: {(patient_counts == 3).sum()}"
)

report_lines.append(
    f"4 ECGs: {(patient_counts == 4).sum()}"
)

report_lines.append(
    f"5+ ECGs: {(patient_counts >= 5).sum()}"
)

report_lines.append(
    f"Maximum ECGs per patient: {patient_counts.max()}"
)

report_lines.append("")

report_lines.append(
    "CONCLUSION:"
)

report_lines.append(
    "Patient-level leakage must be considered when designing"
)

report_lines.append(
    "training, validation, and test splits."
)

OUT_FILE.write_text(
    "\n".join(report_lines),
    encoding="utf-8"
)

print()
print("Report saved to:")
print(OUT_FILE)

print()
print("Experiment 2C completed.")