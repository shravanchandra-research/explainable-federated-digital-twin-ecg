"""Retrospective patient-specific support/query adaptation on held-out patients."""
from __future__ import annotations

import copy
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from .data import PTBXLH5Dataset, read_metadata
from .training import write_json


def _predictions(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            ys.append(y.numpy())
            ps.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)


def _per_class_f1(y_true, y_prob, thresholds):
    pred = (y_prob >= thresholds[None, :]).astype(int)
    return np.asarray([f1_score(y_true[:, k], pred[:, k], zero_division=0)
                       for k in range(y_true.shape[1])], dtype=float)


def evaluate_patient_adaptation(
    global_model, h5_path: str | Path, output_json: str | Path, thresholds=None,
    support_records: int = 1, adaptation_epochs: int = 5, learning_rate: float = 1e-4,
    batch_size: int = 8, device: str = "cpu", evaluation_fold: int = 10, seed: int = 17,
):
    """Adapt a copy of the global model per patient; evaluate only disjoint queries.

    Patients are eligible when at least support_records + 1 ECGs exist in the
    evaluation fold. The global model must have been trained without any of
    these evaluation patients. Records are sorted deterministically by ECG ID;
    this is not a claim of chronological ordering.
    """
    if support_records < 1 or adaptation_epochs < 1:
        raise ValueError("support_records and adaptation_epochs must be positive")
    torch.manual_seed(seed)
    device_t = torch.device(device)
    model_state = copy.deepcopy(global_model.state_dict())
    metadata = read_metadata(h5_path)
    rows = np.flatnonzero(metadata["fold"].astype(int) == int(evaluation_fold))
    patient_rows: dict[int, list[int]] = defaultdict(list)
    for row in rows:
        patient_rows[int(metadata["patient_id"][row])].append(int(row))
    thresholds = np.asarray(thresholds if thresholds is not None else np.full(5, 0.5), dtype=float)

    all_deltas, per_patient = [], []
    for patient_id, records in sorted(patient_rows.items()):
        records = sorted(records, key=lambda r: int(metadata["ecg_id"][r]))
        if len(records) <= support_records:
            continue
        support, query = np.asarray(records[:support_records]), np.asarray(records[support_records:])
        support_ds = PTBXLH5Dataset(h5_path, support)
        query_ds = PTBXLH5Dataset(h5_path, query)
        support_loader = DataLoader(support_ds, batch_size=min(batch_size, len(support)), shuffle=True)
        query_loader = DataLoader(query_ds, batch_size=batch_size, shuffle=False)

        local_model = type(global_model)().to(device_t)
        local_model.load_state_dict(model_state)
        global_y, global_p = _predictions(local_model, query_loader, device_t)
        global_f1 = _per_class_f1(global_y.astype(int), global_p, thresholds)

        optimizer = torch.optim.AdamW(local_model.parameters(), lr=learning_rate, weight_decay=1e-4)
        criterion = torch.nn.BCEWithLogitsLoss()
        for _ in range(adaptation_epochs):
            local_model.train()
            for x, y in support_loader:
                x, y = x.to(device_t), y.to(device_t)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(local_model(x), y)
                loss.backward()
                optimizer.step()

        adapted_y, adapted_p = _predictions(local_model, query_loader, device_t)
        adapted_f1 = _per_class_f1(adapted_y.astype(int), adapted_p, thresholds)
        delta = adapted_f1 - global_f1
        all_deltas.append(delta)
        per_patient.append({"patient_id": patient_id, "support_records": int(len(support)),
                            "query_records": int(len(query)), "global_f1": global_f1.tolist(),
                            "adapted_f1": adapted_f1.tolist(), "delta_f1": delta.tolist()})
        support_ds.close()
        query_ds.close()

    if not all_deltas:
        raise ValueError("No eligible patients in the selected evaluation fold")
    delta_matrix = np.vstack(all_deltas)
    result = {
        "implementation": "new_reference_implementation",
        "evaluation_fold": int(evaluation_fold),
        "support_records_per_patient": int(support_records),
        "adaptation_epochs": int(adaptation_epochs),
        "learning_rate": float(learning_rate),
        "eligible_patients": int(len(per_patient)),
        "class_order": ["NORM", "MI", "STTC", "CD", "HYP"],
        "mean_delta_f1": np.mean(delta_matrix, axis=0).tolist(),
        "std_delta_f1": np.std(delta_matrix, axis=0, ddof=1).tolist() if len(delta_matrix) > 1 else [0.0] * 5,
        "patient_level_results": per_patient,
        "warning": "Exploratory simulation; query records are disjoint from support records. Verify global-model patient exclusion before interpreting results.",
    }
    write_json(result, output_json)
    return result
