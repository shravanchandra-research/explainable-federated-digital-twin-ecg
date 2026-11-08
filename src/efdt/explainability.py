"""Integrated Gradients and simple attribution-audit utilities for temporal ECGs."""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn


def integrated_gradients(model: nn.Module, inputs: torch.Tensor, target_class: int,
                         baseline: torch.Tensor | None = None, n_steps: int = 64) -> torch.Tensor:
    """Compute trapezoidal Integrated Gradients for one or more ECG inputs."""
    if inputs.ndim != 3:
        raise ValueError("inputs must have shape (batch, leads, samples)")
    if n_steps < 2:
        raise ValueError("n_steps must be at least 2")
    baseline = torch.zeros_like(inputs) if baseline is None else baseline.to(inputs.device, inputs.dtype)
    if baseline.shape != inputs.shape:
        raise ValueError("baseline must have the same shape as inputs")
    delta = inputs - baseline
    alphas = torch.linspace(0.0, 1.0, n_steps + 1, device=inputs.device, dtype=inputs.dtype)
    gradient_sum = torch.zeros_like(inputs)
    model_was_training = model.training
    model.eval()
    try:
        for index, alpha in enumerate(alphas):
            point = (baseline + alpha * delta).detach().requires_grad_(True)
            logits = model(point)
            score = logits[:, int(target_class)].sum()
            grads = torch.autograd.grad(score, point, retain_graph=False, create_graph=False)[0]
            weight = 0.5 if index in (0, n_steps) else 1.0
            gradient_sum += weight * grads.detach()
    finally:
        model.train(model_was_training)
    return delta * (gradient_sum / n_steps)


def perturbation_faithfulness(model: nn.Module, inputs: torch.Tensor, target_class: int,
                              attributions: torch.Tensor, fraction: float = 0.1,
                              baseline: torch.Tensor | None = None) -> dict[str, float]:
    """Delete highest-attribution time points across all leads and report score drop."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0,1]")
    baseline = torch.zeros_like(inputs) if baseline is None else baseline.to(inputs.device, inputs.dtype)
    salience = attributions.abs().mean(dim=1)
    n_remove = max(1, int(round(inputs.shape[-1] * fraction)))
    indices = torch.topk(salience, k=n_remove, dim=-1).indices
    perturbed = inputs.clone()
    mask = torch.zeros_like(salience, dtype=torch.bool)
    mask.scatter_(1, indices, True)
    perturbed = torch.where(mask[:, None, :], baseline, perturbed)
    model.eval()
    with torch.no_grad():
        before = torch.sigmoid(model(inputs)[:, target_class]).mean().item()
        after = torch.sigmoid(model(perturbed)[:, target_class]).mean().item()
    return {"original_mean_probability": float(before), "perturbed_mean_probability": float(after),
            "probability_drop": float(before - after), "removed_time_fraction": float(fraction)}


def attribution_stability(model: nn.Module, inputs: torch.Tensor, target_class: int,
                           n_repeats: int = 5, noise_std: float = 0.01,
                           n_steps: int = 32, seed: int = 17) -> float:
    """Mean cosine similarity of noisy-input IG maps to the original IG map."""
    if n_repeats < 1 or noise_std < 0:
        raise ValueError("n_repeats must be >=1 and noise_std must be non-negative")
    generator = torch.Generator(device=inputs.device).manual_seed(seed)
    base_attr = integrated_gradients(model, inputs, target_class, n_steps=n_steps).flatten(1)
    sims = []
    for _ in range(n_repeats):
        noise = torch.randn(inputs.shape, generator=generator, device=inputs.device, dtype=inputs.dtype) * noise_std
        noisy_attr = integrated_gradients(model, inputs + noise, target_class, n_steps=n_steps).flatten(1)
        sims.append(nn.functional.cosine_similarity(base_attr, noisy_attr, dim=1).mean().item())
    return float(np.mean(sims))


def randomized_parameter_sanity(model: nn.Module, inputs: torch.Tensor, target_class: int,
                                n_steps: int = 32, seed: int = 17) -> float:
    """Cosine similarity between trained and parameter-randomized IG maps."""
    torch.manual_seed(seed)
    original = integrated_gradients(model, inputs, target_class, n_steps=n_steps).flatten(1)
    randomized = copy.deepcopy(model)
    for module in randomized.modules():
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            module.reset_parameters()
    randomized_attr = integrated_gradients(randomized, inputs, target_class, n_steps=n_steps).flatten(1)
    return float(nn.functional.cosine_similarity(original, randomized_attr, dim=1).mean().item())
