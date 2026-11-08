"""Analytical model/resource accounting and optional measured inference timing."""
from __future__ import annotations

import time

import torch

from .model import CompactECGCNN, count_parameters


def analytical_resource_profile(throughput_mmac_s: float = 100.0, power_w: float = 0.5) -> dict:
    if throughput_mmac_s <= 0 or power_w < 0:
        raise ValueError("throughput must be positive and power must be non-negative")
    params = 14581
    macs = 672000 + 320000 + 317440 + 320
    weight_fp32 = params * 4
    input_fp32 = 12 * 1000 * 4
    retained_activation_elements = 20037
    activation_fp32 = retained_activation_elements * 4
    bytes_fp32 = weight_fp32 + input_fp32 + activation_fp32
    latency_s = macs / (throughput_mmac_s * 1e6)
    energy_j = power_w * latency_s
    return {
        "trainable_parameters": params,
        "macs_per_ecg": macs,
        "weight_storage_kib_fp32": weight_fp32 / 1024,
        "weight_storage_kib_int8": params / 1024,
        "input_buffer_kib_fp32": input_fp32 / 1024,
        "input_buffer_kib_int8": (12 * 1000) / 1024,
        "retained_intermediate_kib_fp32": activation_fp32 / 1024,
        "retained_intermediate_kib_int8": retained_activation_elements / 1024,
        "conservative_tensor_footprint_kib_fp32": bytes_fp32 / 1024,
        "conservative_tensor_footprint_kib_int8": (params + 12 * 1000 + retained_activation_elements) / 1024,
        "scenario_throughput_mmac_s": throughput_mmac_s,
        "scenario_power_w": power_w,
        "scenario_latency_ms": latency_s * 1000,
        "scenario_energy_mj": energy_j * 1000,
        "measurement_warning": "Latency and energy are assumption-based scenarios, not measured hardware results.",
    }


def measure_inference_latency(model: CompactECGCNN, device: str = "cpu", warmup: int = 20,
                              repetitions: int = 100) -> dict[str, float | str]:
    """Empirically time a local device; output is machine-specific, not wearable evidence."""
    if warmup < 0 or repetitions < 1:
        raise ValueError("warmup must be non-negative and repetitions must be positive")
    device_t = torch.device(device)
    model = model.to(device_t).eval()
    sample = torch.zeros(1, 12, 1000, device=device_t)
    with torch.no_grad():
        for _ in range(warmup):
            model(sample)
        if device_t.type == "cuda":
            torch.cuda.synchronize(device_t)
        start = time.perf_counter()
        for _ in range(repetitions):
            model(sample)
        if device_t.type == "cuda":
            torch.cuda.synchronize(device_t)
        elapsed = time.perf_counter() - start
    return {"device": str(device_t), "warmup_runs": warmup, "timed_runs": repetitions,
            "mean_latency_ms": elapsed * 1000 / repetitions,
            "parameters": count_parameters(model),
            "warning": "Measured only on the selected host device and software stack; not a wearable-device measurement."}
