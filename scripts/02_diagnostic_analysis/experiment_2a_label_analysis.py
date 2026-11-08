import os
import ast
import pandas as pd

# ============================================================
# EXPERIMENT 2A
# PTB-XL DIAGNOSTIC LABEL ANALYSIS
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

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "experiments",
    "experiment_2a_label_analysis.txt"
)

print("=" * 70)
print("EXPERIMENT 2A - PTB-XL DIAGNOSTIC LABEL ANALYSIS")
print("=" * 70)

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

df = pd.read_csv(DATABASE_FILE)
scp_df = pd.read_csv(SCP_FILE)

print("\nTotal ECG records:", len(df))
print("Unique patients:", df["patient_id"].nunique())

# ------------------------------------------------------------
# Identify diagnostic SCP codes
# ------------------------------------------------------------

diagnostic_codes = set(
    scp_df.loc[
        scp_df["diagnostic"] == 1,
        "Unnamed: 0"
    ]
)

print(
    "\nNumber of diagnostic SCP codes:",
    len(diagnostic_codes)
)

# ------------------------------------------------------------
# Parse SCP codes for each ECG
# ------------------------------------------------------------

def get_diagnostic_codes(value):

    try:
        codes = ast.literal_eval(value)
    except Exception:
        return []

    return [
        code
        for code in codes.keys()
        if code in diagnostic_codes
    ]


df["diagnostic_codes"] = (
    df["scp_codes"]
    .apply(get_diagnostic_codes)
)

# ------------------------------------------------------------
# Diagnostic code frequency
# ------------------------------------------------------------

code_counts = {}

for codes in df["diagnostic_codes"]:

    for code in codes:

        code_counts[code] = (
            code_counts.get(code, 0) + 1
        )

code_counts_df = (
    pd.DataFrame(
        list(code_counts.items()),
        columns=["Code", "Count"]
    )
    .sort_values(
        "Count",
        ascending=False
    )
)

print("\n--- TOP DIAGNOSTIC CODES ---")

print(
    code_counts_df.head(30).to_string(
        index=False
    )
)

# ------------------------------------------------------------
# Diagnostic superclass information
# ------------------------------------------------------------

superclass_columns = [
    "diagnostic_class"
]

available_columns = [
    c for c in superclass_columns
    if c in scp_df.columns
]

print("\nAvailable SCP columns:")

for column in scp_df.columns:
    print("-", column)

# ------------------------------------------------------------
# Show diagnostic codes and their metadata
# ------------------------------------------------------------

print("\n--- DIAGNOSTIC CODE METADATA ---")

diagnostic_table = scp_df[
    scp_df["Unnamed: 0"].isin(
        diagnostic_codes
    )
]

print(
    diagnostic_table[
        [
            c for c in [
                "Unnamed: 0",
                "description",
                "diagnostic",
                "diagnostic_class"
            ]
            if c in diagnostic_table.columns
        ]
    ].head(50).to_string(index=False)
)

# ------------------------------------------------------------
# Fold distribution
# ------------------------------------------------------------

print("\n--- FOLD DISTRIBUTION ---")

fold_counts = (
    df["strat_fold"]
    .value_counts()
    .sort_index()
)

print(fold_counts)

# ------------------------------------------------------------
# Patient-level counts
# ------------------------------------------------------------

print("\n--- PATIENT INFORMATION ---")

print(
    "Total patients:",
    df["patient_id"].nunique()
)

print(
    "Patients with multiple ECGs:",
    (
        df.groupby("patient_id")
        .size()
        .gt(1)
        .sum()
    )
)

# ------------------------------------------------------------
# Save report
# ------------------------------------------------------------

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "EXPERIMENT 2A - PTB-XL DIAGNOSTIC LABEL ANALYSIS\n"
    )

    f.write("=" * 60 + "\n\n")

    f.write(
        f"Total ECG records: {len(df)}\n"
    )

    f.write(
        f"Unique patients: {df['patient_id'].nunique()}\n"
    )

    f.write(
        f"Diagnostic SCP codes: {len(diagnostic_codes)}\n"
    )

    f.write("\nTop diagnostic codes:\n")

    f.write(
        code_counts_df.head(50)
        .to_string(index=False)
    )

    f.write("\n\nFold distribution:\n")

    f.write(
        fold_counts.to_string()
    )

print("\nReport saved to:")
print(OUTPUT_FILE)

print("\nExperiment 2A completed.")