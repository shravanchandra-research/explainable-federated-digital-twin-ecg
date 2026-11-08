"""Retrospective, data-driven patient-state replay (not a live physiological twin)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import json

import numpy as np
import torch


@dataclass
class PatientTwinState:
    patient_id: int
    observation_index: int = 0
    latest_ecg_id: int | None = None
    embedding: list[float] = field(default_factory=list)
    class_probabilities: list[float] = field(default_factory=list)
    signal_quality: dict[str, float] = field(default_factory=dict)
    adaptation_status: str = "not_adapted"
    model_version: str = "unspecified"
    last_updated_utc: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "PatientTwinState":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)


def replay_patient_observation(model, state: PatientTwinState, ecg: torch.Tensor,
                               ecg_id: int, model_version: str,
                               embedding_ema: float = 0.2,
                               device: str = "cpu") -> PatientTwinState:
    """Advance a patient's retrospective state by one already-recorded ECG."""
    if not 0 < embedding_ema <= 1:
        raise ValueError("embedding_ema must be in (0,1]")
    if ecg.ndim == 2:
        ecg = ecg.unsqueeze(0)
    if ecg.shape != (1, 12, 1000):
        raise ValueError(f"Expected one ECG with shape (1,12,1000), received {tuple(ecg.shape)}")
    if not torch.isfinite(ecg).all():
        raise ValueError("ECG contains NaN or infinite values")
    device_t = torch.device(device)
    model = model.to(device_t).eval()
    ecg = ecg.to(device_t, dtype=torch.float32)
    with torch.no_grad():
        embedding = model.extract_features(ecg)[0].cpu().numpy().astype(float)
        probabilities = torch.sigmoid(model(ecg))[0].cpu().numpy().astype(float)
    signal = ecg[0].detach().cpu().numpy()
    quality = {
        "finite_fraction": float(np.isfinite(signal).mean()),
        "mean_absolute_amplitude": float(np.mean(np.abs(signal))),
        "mean_lead_standard_deviation": float(np.mean(np.std(signal, axis=1))),
        "near_zero_lead_fraction": float(np.mean(np.std(signal, axis=1) < 1e-6)),
    }
    if state.embedding:
        previous = np.asarray(state.embedding, dtype=float)
        if previous.shape != embedding.shape:
            raise ValueError("Stored embedding dimension does not match the current model")
        embedding = (1 - embedding_ema) * previous + embedding_ema * embedding
    state.embedding = embedding.tolist()
    state.class_probabilities = probabilities.tolist()
    state.signal_quality = quality
    state.latest_ecg_id = int(ecg_id)
    state.observation_index += 1
    state.model_version = str(model_version)
    state.last_updated_utc = datetime.now(timezone.utc).isoformat()
    return state
