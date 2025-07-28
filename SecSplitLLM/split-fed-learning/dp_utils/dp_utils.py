import torch
from typing import List
from opacus.accountants import RDPAccountant

GLOBAL_NORMS = []
GLOBAL_SERVER_GRAD_NORMS = []

def clip_and_add_noise(
    activations: torch.Tensor,
    clip_threshold: float,
    noise_multiplier: float
) -> torch.Tensor:
    # Flatten activations per sample
    batch_size = activations.size(0)
    # flat_activations = activations.view(batch_size, -1)  # shape: (batch_size, hidden_dim * seq_len)
    flat_activations = activations if activations.dim() == 2 else activations.view(batch_size, -1)

    # Compute L2 norm for each sample
    l2_norms = flat_activations.norm(p=2, dim=1, keepdim=True)  # shape: (batch_size, 1)


    # GLOBAL_NORMS.extend(l2_norms.squeeze().tolist())

    # Clip
    scaling_factors = (clip_threshold / (l2_norms + 1e-6)).clamp(max=1.0)
    clipped_flat = flat_activations * scaling_factors

    clipped_activations = clipped_flat if activations.dim() == 2 else clipped_flat.view_as(activations)

    # Add Gaussian noise
    noise_std = clip_threshold * noise_multiplier
    noise = torch.randn_like(clipped_activations) * noise_std

    return clipped_activations + noise


class DPAccountant:
    # Differential Privacy Accountant using Opacus RDPAccountant.
    # Tracks epsilon over training steps for given noise multiplier and sample rate.
    def __init__(self, noise_multiplier: float, sample_rate: float, steps: int = 0):
        self.noise_multiplier = noise_multiplier
        self.sample_rate = sample_rate
        self.steps = steps
        self.accountant = RDPAccountant()

        for _ in range(self.steps):
            self.accountant.step(
                noise_multiplier=self.noise_multiplier,
                sample_rate=self.sample_rate
            )

    def step(self):
        self.accountant.step(
            noise_multiplier=self.noise_multiplier,
            sample_rate=self.sample_rate
        )
        self.steps += 1

    def get_epsilon(self, delta: float):
        # Returns (ε, optimal_order) for the given δ using RDP accounting.
        return self.accountant.get_epsilon(delta)


def clip_gradients(grad_batch: torch.Tensor, max_norm: float = 1.0):
    # Clips per-example gradients to a maximum L2 norm.
    batch_size = grad_batch.shape[0]
    grad_flat = grad_batch.view(batch_size, -1)
    norms = torch.norm(grad_flat, p=2, dim=1, keepdim=True)
    # GLOBAL_SERVER_GRAD_NORMS.extend(norms.squeeze().cpu().tolist())

    clip_factors = (max_norm / (norms + 1e-6)).clamp(max=1.0)
    grad_clipped = grad_flat * clip_factors
        
    return grad_clipped.view_as(grad_batch)


def add_noise(grad_batch: torch.Tensor, noise_multiplier: float, max_norm: float):
    # Adds Gaussian noise to gradients after clipping.
    noise_std = noise_multiplier * max_norm
    noise = torch.randn_like(grad_batch) * noise_std
    return grad_batch + noise
