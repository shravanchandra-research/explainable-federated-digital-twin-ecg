"""HDF5 reader for the Experiment 5A PTB-XL output format."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

LABELS = ("NORM", "MI", "STTC", "CD", "HYP")


class PTBXLH5Dataset(Dataset):
    """Lazy HDF5 dataset; converts stored (time, leads) to model (leads, time)."""

    def __init__(self, h5_path: str | Path, row_indices: Sequence[int] | None = None):
        self.h5_path = str(h5_path)
        with h5py.File(self.h5_path, "r") as h5:
            required = ("X", "Y", "ecg_id", "patient_id", "fold")
            missing = [k for k in required if k not in h5]
            if missing:
                raise KeyError(f"HDF5 file is missing required datasets: {missing}")
            n = len(h5["Y"])
            if row_indices is None:
                self.row_indices = np.arange(n, dtype=np.int64)
            else:
                self.row_indices = np.asarray(row_indices, dtype=np.int64)
                if np.any(self.row_indices < 0) or np.any(self.row_indices >= n):
                    raise IndexError("row_indices contains an out-of-range HDF5 row")
            if h5["X"].shape[1:] != (1000, 12):
                raise ValueError(f"Expected HDF5 X shape (N,1000,12), got {h5['X'].shape}")
            if h5["Y"].shape[1:] != (5,):
                raise ValueError(f"Expected HDF5 Y shape (N,5), got {h5['Y'].shape}")
        self._h5 = None

    def __len__(self) -> int:
        return int(self.row_indices.size)

    def _file(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __getitem__(self, item: int):
        row = int(self.row_indices[item])
        h5 = self._file()
        signal = np.asarray(h5["X"][row], dtype=np.float32).T.copy()
        target = np.asarray(h5["Y"][row], dtype=np.float32)
        return torch.from_numpy(signal), torch.from_numpy(target)

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5"] = None
        return state


def read_metadata(h5_path: str | Path) -> dict[str, np.ndarray]:
    """Read small metadata arrays only; do not load the waveform tensor."""
    with h5py.File(h5_path, "r") as h5:
        result = {key: np.asarray(h5[key]) for key in ("ecg_id", "patient_id", "fold", "Y")}
    return result


def fold_rows(metadata: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    folds = metadata["fold"].astype(int)
    train = np.flatnonzero(np.isin(folds, np.arange(1, 9)))
    validation = np.flatnonzero(folds == 9)
    test = np.flatnonzero(folds == 10)
    if not train.size or not validation.size or not test.size:
        raise ValueError("Expected non-empty PTB-XL folds 1-8 (train), 9 (validation), and 10 (test)")
    return train, validation, test
