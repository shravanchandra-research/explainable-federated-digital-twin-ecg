import os
import ast
import pandas as pd

# ============================================================
# EXPERIMENT 2B
# PTB-XL DIAGNOSTIC SUPERCLASS ANALYSIS
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
    "experiment_2b_superclass_analysis.txt"
)

print("=" * 70)
print("EXPERIMENT 2B - PTB-XL DIAGNOSTIC SUPERCLASS ANALYSIS")
print("=" * 70)

# ------------------------------------------------------------
# Load files
# ------------------------------------------------------------

df = pd.read_csv(DATABASE_FILE)
scp = pd.read_csv(SCP_FILE)

print("\nTotal ECG records :", len(df))
print("Unique patients   :", df["patient_id"].nunique())

# ------------------------------------------------------------
# Identify diagnostic SCP statements
# ------------------------------------------------------------

diagnostic_scp = scp[
    scp["diagnostic"] == 1
].copy()

code_to_class = dict(
    zip(
        diagnostic_scp["Unnamed: 0"],
        diagnostic_scp["diagnostic_class"]
    )
)

# ------------------------------------------------------------
# Extract diagnostic classes
# ------------------------------------------------------------

def extract_classes(value):

    try:
        codes = ast.literal_eval(value)
    except Exception:
        return []

    classes = set()

    for code in codes.keys():

        if code in code_to_class:

            diagnostic_class = code_to_class[code]

            if pd.notna(diagnostic_class):
                classes.add(
                    diagnostic_class
                )

    return sorted(classes)


df["diagnostic_classes"] = (
    df["scp_codes"]
    .apply(extract_classes)
)

# ------------------------------------------------------------
# Count ECGs per superclass
# ------------------------------------------------------------

superclasses = [
    "NORM",
    "MI",
    "STTC",
    "CD",
    "HYP"
]

counts = {}

for superclass in superclasses:

    counts[superclass] = (
        df["diagnostic_classes"]
        .apply(
            lambda x:
            superclass in x
        )
        .sum()
    )

print("\n--- DIAGNOSTIC SUPERCLASS COUNTS ---")

for superclass, count in counts.items():

    percentage = (
        count / len(df)
    ) * 100

    print(
        f"{superclass:6s} : "
        f"{count:6d} "
        f"({percentage:.2f}%)"
    )

# ------------------------------------------------------------
# Multi-label statistics
# ------------------------------------------------------------

label_counts = (
    df["diagnostic_classes"]
    .apply(len)
)

print("\n--- MULTI-LABEL STRUCTURE ---")

print(
    "ECGs with zero diagnostic classes:",
    (label_counts == 0).sum()
)

print(
    "ECGs with one diagnostic class:",
    (label_counts == 1).sum()
)

print(
    "ECGs with two diagnostic classes:",
    (label_counts == 2).sum()
)

print(
    "ECGs with three or more classes:",
    (label_counts >= 3).sum()
)

print(
    "Maximum classes in one ECG:",
    label_counts.max()
)

# ------------------------------------------------------------
# Class combinations
# ------------------------------------------------------------

combination_counts = (
    df["diagnostic_classes"]
    .apply(
        lambda x:
        "+".join(x) if x else "NONE"
    )
    .value_counts()
)

print("\n--- TOP CLASS COMBINATIONS ---")

print(
    combination_counts.head(20)
)

# ------------------------------------------------------------
# Fold-wise superclass distribution
# ------------------------------------------------------------

print("\n--- FOLD-WISE DISTRIBUTION ---")

fold_table = []

for fold in sorted(
    df["strat_fold"].unique()
):

    subset = df[
        df["strat_fold"] == fold
    ]

    row = {
        "fold": fold,
        "records": len(subset)
    }

    for superclass in superclasses:

        row[superclass] = (
            subset["diagnostic_classes"]
            .apply(
                lambda x:
                superclass in x
            )
            .sum()
        )

    fold_table.append(row)

fold_df = pd.DataFrame(fold_table)

print(
    fold_df.to_string(index=False)
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
        "EXPERIMENT 2B - PTB-XL DIAGNOSTIC SUPERCLASS ANALYSIS\n"
    )

    f.write("=" * 60 + "\n\n")

    f.write(
        f"Total ECG records: {len(df)}\n"
    )

    f.write(
        f"Unique patients: {df['patient_id'].nunique()}\n\n"
    )

    f.write(
        "Superclass counts:\n"
    )

    for superclass, count in counts.items():

        percentage = (
            count / len(df)
        ) * 100

        f.write(
            f"{superclass}: "
            f"{count} "
            f"({percentage:.2f}%)\n"
        )

    f.write(
        "\nMulti-label statistics:\n"
    )

    f.write(
        f"Zero classes: {(label_counts == 0).sum()}\n"
    )

    f.write(
        f"One class: {(label_counts == 1).sum()}\n"
    )

    f.write(
        f"Two classes: {(label_counts == 2).sum()}\n"
    )

    f.write(
        f"Three or more: {(label_counts >= 3).sum()}\n"
    )

    f.write(
        f"Maximum classes: {label_counts.max()}\n"
    )

    f.write(
        "\nTop combinations:\n"
    )

    f.write(
        combination_counts.head(20)
        .to_string()
    )

    f.write(
        "\n\nFold distribution:\n"
    )

    f.write(
        fold_df.to_string(index=False)
    )

print("\nReport saved to:")
print(OUTPUT_FILE)

print("\nExperiment 2B completed.")