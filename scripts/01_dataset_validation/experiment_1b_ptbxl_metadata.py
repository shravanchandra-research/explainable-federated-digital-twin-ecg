import os
import ast
import pandas as pd

# ============================================================
# EXPERIMENT 1B
# PTB-XL METADATA VERIFICATION
# ============================================================

BASE_DIR = os.environ.get("PTBXL_ROOT", r"C:\PTB-XL")

DATABASE_FILE = os.path.join(
    BASE_DIR,
    "ptbxl_database.csv"
)

SCP_FILE = os.path.join(
    BASE_DIR,
    "scp_statements.csv"
)

RECORD_ID = "01000"

print("=" * 70)
print("EXPERIMENT 1B - PTB-XL METADATA VERIFICATION")
print("=" * 70)

# ------------------------------------------------------------
# Load PTB-XL database
# ------------------------------------------------------------

print("\nLoading:")
print(DATABASE_FILE)

df = pd.read_csv(DATABASE_FILE)

print("\nTotal metadata records:", len(df))

# ------------------------------------------------------------
# Locate our ECG record
# ------------------------------------------------------------

record = df[df["ecg_id"] == int(RECORD_ID)]

if record.empty:
    print("\nERROR: Record not found.")
    raise SystemExit(1)

row = record.iloc[0]

# ------------------------------------------------------------
# Display core metadata
# ------------------------------------------------------------

print("\n--- RECORD METADATA ---")

print("ECG ID       :", row["ecg_id"])
print("Patient ID   :", row["patient_id"])
print("Filename     :", row["filename_lr"])
print("Age          :", row["age"])
print("Sex          :", row["sex"])
print("Height       :", row["height"])
print("Weight       :", row["weight"])

print("\n--- DIAGNOSTIC INFORMATION ---")

print("Diagnostic statements:")
print(row["scp_codes"])

print("\n--- FOLD INFORMATION ---")

print("Recommended fold:", row["strat_fold"])

# ------------------------------------------------------------
# Parse SCP codes
# ------------------------------------------------------------

try:
    scp_codes = ast.literal_eval(row["scp_codes"])
except Exception:
    scp_codes = {}

print("\n--- PARSED SCP CODES ---")

for code, value in scp_codes.items():
    print(f"{code}: {value}")

# ------------------------------------------------------------
# Load SCP statements
# ------------------------------------------------------------

scp_df = pd.read_csv(SCP_FILE)

print("\n--- SCP STATEMENT TABLE ---")

print("Number of SCP statements:", len(scp_df))

# ------------------------------------------------------------
# Match diagnostic codes
# ------------------------------------------------------------

print("\n--- CODE DESCRIPTIONS ---")

for code in scp_codes.keys():

    match = scp_df[scp_df["Unnamed: 0"] == code]

    if not match.empty:

        description = match.iloc[0].get(
            "description",
            "Description unavailable"
        )

        diagnostic = match.iloc[0].get(
            "diagnostic",
            None
        )

        print(
            f"{code}: "
            f"{description} | "
            f"diagnostic={diagnostic}"
        )

# ------------------------------------------------------------
# Dataset-level statistics
# ------------------------------------------------------------

print("\n--- DATASET STATISTICS ---")

print("Total ECG records:", len(df))
print(
    "Unique patients:",
    df["patient_id"].nunique()
)

print(
    "Number of folds:",
    df["strat_fold"].nunique()
)

print(
    "Fold distribution:"
)

print(
    df["strat_fold"]
    .value_counts()
    .sort_index()
)

# ------------------------------------------------------------
# Verification
# ------------------------------------------------------------

print("\n--- VERIFICATION ---")

checks = {}

checks["Record exists"] = not record.empty
checks["Patient ID exists"] = pd.notna(row["patient_id"])
checks["Waveform filename exists"] = pd.notna(row["filename_lr"])
checks["SCP codes available"] = pd.notna(row["scp_codes"])
checks["Fold available"] = pd.notna(row["strat_fold"])

for name, result in checks.items():

    print(
        f"{name:30s}: "
        f"{'PASS' if result else 'FAIL'}"
    )

overall = all(checks.values())

print(
    "\nOVERALL METADATA VERIFICATION:",
    "PASS" if overall else "FAIL"
)

# ------------------------------------------------------------
# Save metadata report
# ------------------------------------------------------------

report_file = os.path.join(
    BASE_DIR,
    "experiments",
    "experiment_1b_metadata.txt"
)

with open(
    report_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "EXPERIMENT 1B - PTB-XL METADATA VERIFICATION\n"
    )

    f.write("=" * 60 + "\n\n")

    f.write(
        f"ECG ID: {row['ecg_id']}\n"
    )

    f.write(
        f"Patient ID: {row['patient_id']}\n"
    )

    f.write(
        f"Filename: {row['filename_lr']}\n"
    )

    f.write(
        f"Age: {row['age']}\n"
    )

    f.write(
        f"Sex: {row['sex']}\n"
    )

    f.write(
        f"SCP codes: {row['scp_codes']}\n"
    )

    f.write(
        f"Recommended fold: {row['strat_fold']}\n"
    )

    f.write(
        f"\nTotal ECG records: {len(df)}\n"
    )

    f.write(
        f"Unique patients: {df['patient_id'].nunique()}\n"
    )

    f.write(
        f"Number of folds: {df['strat_fold'].nunique()}\n"
    )

    f.write(
        "\nVERIFICATION: "
        + ("PASS" if overall else "FAIL")
        + "\n"
    )

print("\nReport saved to:")
print(report_file)

print("\nExperiment 1B completed.")