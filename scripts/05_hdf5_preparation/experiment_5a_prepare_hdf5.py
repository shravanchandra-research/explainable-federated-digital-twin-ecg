import os
import sys
import numpy as np
import pandas as pd
import h5py

# ============================================================
# EXPERIMENT 5A
# PTB-XL MODEL-READY DATASET PREPARATION
# Convert 4H preprocessed NPY files into HDF5
# ============================================================

BASE_DIR = os.environ.get("PTBXL_ROOT", r"C:\PTB-XL")

INDEX_FILE = os.path.join(
    BASE_DIR,
    "experiments",
    "4H_output",
    "experiment_4h_full_index.csv"
)

OUTPUT_H5 = os.path.join(
    BASE_DIR,
    "experiments",
    "4H_output",
    "ptbxl_4h_model_ready.h5"
)

ERROR_FILE = os.path.join(
    BASE_DIR,
    "experiments",
    "4H_output",
    "experiment_5a_hdf5_errors.csv"
)

# ============================================================
# DATASET CONFIGURATION
# ============================================================

NUM_RECORDS = 21799
NUM_SAMPLES = 1000
NUM_LEADS = 12
NUM_CLASSES = 5

LABEL_COLUMNS = [
    "NORM",
    "MI",
    "STTC",
    "CD",
    "HYP"
]

DTYPE_X = np.float32
DTYPE_Y = np.int8

# ============================================================
# START
# ============================================================

print("=" * 70)
print("EXPERIMENT 5A - PTB-XL HDF5 MODEL-READY DATASET")
print("=" * 70)

# ============================================================
# CHECK INDEX FILE
# ============================================================

print()
print("Loading model-ready index...")

if not os.path.exists(INDEX_FILE):
    print("ERROR: Index file not found:")
    print(INDEX_FILE)
    sys.exit(1)

df = pd.read_csv(INDEX_FILE)

print(f"Records: {len(df)}")

print()
print("Columns:")
print(list(df.columns))

# ============================================================
# REQUIRED COLUMN VALIDATION
# IMPORTANT:
# 4H uses 'fold', not 'strat_fold'
# 4H uses 'validation', not 'val'
# ============================================================

required_columns = [
    "ecg_id",
    "patient_id",
    "fold",
    "split",
    "processed_file"
]

for column in required_columns:

    if column not in df.columns:

        print()
        print(f"ERROR: Missing required column: {column}")
        sys.exit(1)

for column in LABEL_COLUMNS:

    if column not in df.columns:

        print()
        print(f"ERROR: Missing label column: {column}")
        sys.exit(1)

# ============================================================
# RECORD COUNT VALIDATION
# ============================================================

if len(df) != NUM_RECORDS:

    print()
    print(
        f"WARNING: Expected {NUM_RECORDS} records "
        f"but found {len(df)} records."
    )

# ============================================================
# VERIFY SOURCE SPLIT DISTRIBUTION
# ============================================================

print()
print("Source split distribution:")

source_split_counts = df["split"].value_counts(
    dropna=False
)

for split_name, count in source_split_counts.items():

    print(
        f"{str(split_name):12s}: {int(count)}"
    )

# ============================================================
# CREATE / REPLACE HDF5
# ============================================================

print()
print("Creating HDF5 dataset...")
print(f"Output: {OUTPUT_H5}")

if os.path.exists(OUTPUT_H5):

    print()
    print("Existing HDF5 file found.")
    print("It will be replaced.")

    try:

        os.remove(OUTPUT_H5)

    except PermissionError:

        print()
        print("ERROR: HDF5 file is currently open or locked.")
        print(
            "Close any program using the HDF5 file and run again."
        )
        sys.exit(1)

# ============================================================
# PROCESSING VARIABLES
# ============================================================

processed = 0
failed = 0

failure_records = []

# ============================================================
# CREATE HDF5
# ============================================================

with h5py.File(OUTPUT_H5, "w") as h5:

    # --------------------------------------------------------
    # ECG waveform dataset
    # Shape = (21799, 1000, 12)
    # --------------------------------------------------------

    X = h5.create_dataset(
        "X",
        shape=(
            NUM_RECORDS,
            NUM_SAMPLES,
            NUM_LEADS
        ),
        dtype=DTYPE_X,
        compression="gzip",
        compression_opts=4,
        chunks=(1, NUM_SAMPLES, NUM_LEADS)
    )

    # --------------------------------------------------------
    # Multi-label target dataset
    # Shape = (21799, 5)
    # --------------------------------------------------------

    Y = h5.create_dataset(
        "Y",
        shape=(
            NUM_RECORDS,
            NUM_CLASSES
        ),
        dtype=DTYPE_Y
    )

    # --------------------------------------------------------
    # ECG ID
    # --------------------------------------------------------

    ECG_ID = h5.create_dataset(
        "ecg_id",
        shape=(NUM_RECORDS,),
        dtype=np.int64
    )

    # --------------------------------------------------------
    # Patient ID
    # --------------------------------------------------------

    PATIENT_ID = h5.create_dataset(
        "patient_id",
        shape=(NUM_RECORDS,),
        dtype=np.int64
    )

    # --------------------------------------------------------
    # Official PTB-XL fold
    # --------------------------------------------------------

    FOLD = h5.create_dataset(
        "fold",
        shape=(NUM_RECORDS,),
        dtype=np.int8
    )

    # --------------------------------------------------------
    # Dataset split
    # Values:
    # train
    # validation
    # test
    # --------------------------------------------------------

    SPLIT = h5.create_dataset(
        "split",
        shape=(NUM_RECORDS,),
        dtype=h5py.string_dtype(
            encoding="utf-8"
        )
    )

    # --------------------------------------------------------
    # HDF5 ATTRIBUTES
    # --------------------------------------------------------

    h5.attrs["dataset"] = "PTB-XL"

    h5.attrs["version"] = "1.0.3"

    h5.attrs["sampling_rate"] = 100

    h5.attrs["num_leads"] = NUM_LEADS

    h5.attrs["num_samples"] = NUM_SAMPLES

    h5.attrs["num_classes"] = NUM_CLASSES

    h5.attrs["label_order"] = ",".join(
        LABEL_COLUMNS
    )

    # --------------------------------------------------------
    # PROCESS EVERY ECG
    # --------------------------------------------------------

    for i, row in df.iterrows():

        try:

            # ------------------------------------------------
            # Processed NPY path
            # ------------------------------------------------

            processed_file = str(
                row["processed_file"]
            )

            if not os.path.isabs(
                processed_file
            ):

                processed_file = os.path.join(
                    BASE_DIR,
                    processed_file
                )

            processed_file = os.path.normpath(
                processed_file
            )

            # ------------------------------------------------
            # Check file
            # ------------------------------------------------

            if not os.path.exists(
                processed_file
            ):

                raise FileNotFoundError(
                    f"Processed file not found: "
                    f"{processed_file}"
                )

            # ------------------------------------------------
            # Load waveform
            # ------------------------------------------------

            signal = np.load(
                processed_file
            )

            # ------------------------------------------------
            # Validate waveform shape
            # ------------------------------------------------

            if signal.shape != (
                NUM_SAMPLES,
                NUM_LEADS
            ):

                raise ValueError(
                    f"Invalid waveform shape: "
                    f"{signal.shape}"
                )

            # ------------------------------------------------
            # Validate finite values
            # ------------------------------------------------

            if not np.isfinite(
                signal
            ).all():

                raise ValueError(
                    "Waveform contains NaN or Inf"
                )

            # ------------------------------------------------
            # Convert to float32
            # ------------------------------------------------

            signal = signal.astype(
                DTYPE_X,
                copy=False
            )

            # ------------------------------------------------
            # Extract labels
            # ------------------------------------------------

            labels = np.array(
                [
                    int(row[label])
                    for label in LABEL_COLUMNS
                ],
                dtype=DTYPE_Y
            )

            # ------------------------------------------------
            # Validate labels
            # ------------------------------------------------

            if not np.all(
                np.isin(
                    labels,
                    [0, 1]
                )
            ):

                raise ValueError(
                    f"Invalid label values: {labels}"
                )

            # ------------------------------------------------
            # Extract metadata
            # ------------------------------------------------

            ecg_id = int(
                row["ecg_id"]
            )

            patient_id = int(
                row["patient_id"]
            )

            # IMPORTANT:
            # 4H index column = fold
            fold = int(
                row["fold"]
            )

            split = str(
                row["split"]
            ).strip().lower()

            # ------------------------------------------------
            # Validate split
            # ------------------------------------------------

            valid_splits = [
                "train",
                "validation",
                "test"
            ]

            if split not in valid_splits:

                raise ValueError(
                    f"Invalid split: {split}"
                )

            # ------------------------------------------------
            # Write waveform
            # ------------------------------------------------

            X[i] = signal

            # ------------------------------------------------
            # Write labels
            # ------------------------------------------------

            Y[i] = labels

            # ------------------------------------------------
            # Write metadata
            # ------------------------------------------------

            ECG_ID[i] = ecg_id

            PATIENT_ID[i] = patient_id

            FOLD[i] = fold

            SPLIT[i] = split

            processed += 1

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if processed % 500 == 0:

                print(
                    f"Processed: "
                    f"{processed}/{NUM_RECORDS}"
                )

        except Exception as e:

            failed += 1

            failure_records.append(
                {
                    "row": i,
                    "ecg_id": row.get(
                        "ecg_id",
                        "UNKNOWN"
                    ),
                    "error": str(e)
                }
            )

            print(
                f"ERROR at row {i}: {e}"
            )

# ============================================================
# SAVE ERROR REPORT
# ============================================================

if failure_records:

    error_df = pd.DataFrame(
        failure_records
    )

    error_df.to_csv(
        ERROR_FILE,
        index=False
    )

else:

    pd.DataFrame(
        columns=[
            "row",
            "ecg_id",
            "error"
        ]
    ).to_csv(
        ERROR_FILE,
        index=False
    )

# ============================================================
# HDF5 VALIDATION
# ============================================================

print()
print("=" * 70)
print("HDF5 VALIDATION")
print("=" * 70)

if not os.path.exists(
    OUTPUT_H5
):

    print(
        "ERROR: HDF5 file was not created."
    )

    sys.exit(1)

with h5py.File(
    OUTPUT_H5,
    "r"
) as h5:

    # --------------------------------------------------------
    # Dataset list
    # --------------------------------------------------------

    print()
    print("HDF5 datasets:")

    for key in h5.keys():

        print(
            f"  {key}: "
            f"{h5[key].shape}"
        )

    # --------------------------------------------------------
    # Shapes
    # --------------------------------------------------------

    X_shape = h5["X"].shape

    Y_shape = h5["Y"].shape

    print()
    print(
        "X shape:",
        X_shape
    )

    print(
        "Y shape:",
        Y_shape
    )

    # --------------------------------------------------------
    # Validate X
    # --------------------------------------------------------

    if X_shape != (
        NUM_RECORDS,
        NUM_SAMPLES,
        NUM_LEADS
    ):

        print(
            "ERROR: Invalid X shape."
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Validate Y
    # --------------------------------------------------------

    if Y_shape != (
        NUM_RECORDS,
        NUM_CLASSES
    ):

        print(
            "ERROR: Invalid Y shape."
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Load labels
    # --------------------------------------------------------

    Y_data = h5["Y"][:]

    print()
    print(
        "Diagnostic superclass counts:"
    )

    for j, label in enumerate(
        LABEL_COLUMNS
    ):

        count = int(
            np.sum(
                Y_data[:, j]
            )
        )

        print(
            f"{label:5s}: {count}"
        )

    # --------------------------------------------------------
    # Load split
    # --------------------------------------------------------

    split_data = h5["split"][:]

    split_data = [
        s.decode("utf-8")
        if isinstance(s, bytes)
        else str(s)
        for s in split_data
    ]

    # --------------------------------------------------------
    # Split counts
    # IMPORTANT:
    # validation, NOT val
    # --------------------------------------------------------

    train_count = split_data.count(
        "train"
    )

    validation_count = split_data.count(
        "validation"
    )

    test_count = split_data.count(
        "test"
    )

    print()
    print(
        "Dataset split:"
    )

    print(
        "Training   :",
        train_count
    )

    print(
        "Validation :",
        validation_count
    )

    print(
        "Test       :",
        test_count
    )

    # --------------------------------------------------------
    # Validate split totals
    # --------------------------------------------------------

    split_total = (
        train_count
        + validation_count
        + test_count
    )

    print()
    print(
        "Split total :",
        split_total
    )

    if split_total != NUM_RECORDS:

        print()
        print(
            "ERROR: Split counts do not "
            "sum to total records."
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Validate expected PTB-XL split
    # --------------------------------------------------------

    expected_train = 17418

    expected_validation = 2183

    expected_test = 2198

    split_correct = (
        train_count == expected_train
        and
        validation_count == expected_validation
        and
        test_count == expected_test
    )

    print()

    if split_correct:

        print(
            "Official split distribution: PASS"
        )

    else:

        print(
            "Official split distribution: FAIL"
        )

        sys.exit(1)

# ============================================================
# FINAL RESULT
# ============================================================

print()
print("=" * 70)
print(
    "EXPERIMENT 5A - DATA PREPARATION RESULT"
)
print("=" * 70)

print()

print(
    f"Expected records : {NUM_RECORDS}"
)

print(
    f"Processed         : {processed}"
)

print(
    f"Failed            : {failed}"
)

print()

print(
    "HDF5 file:"
)

print(
    OUTPUT_H5
)

print()

print(
    "Error report:"
)

print(
    ERROR_FILE
)

# ============================================================
# FINAL PASS / FAIL
# ============================================================

if (
    processed == NUM_RECORDS
    and failed == 0
    and train_count == 17418
    and validation_count == 2183
    and test_count == 2198
):

    print()

    print(
        "EXPERIMENT 5A DATA PREPARATION RESULT : PASS"
    )

else:

    print()

    print(
        "EXPERIMENT 5A DATA PREPARATION RESULT : FAIL"
    )

print()

print("=" * 70)