import numpy as np
from opacus.accountants import RDPAccountant

# Your client dataset sizes
client_dataset_sizes = [6545, 6545, 6545, 3273, 3273]

# Set your batch size here
batch_size = 8  # change this to your actual batch size

# Privacy parameters
noise_multipliers = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
total_training_steps = 160  # e.g., 10 rounds * 2 epochs * 8 batches (adjust if needed)
delta = 1e-5

def get_epsilon(sample_rate, noise_multiplier, steps, delta):
    if noise_multiplier == 0.0:
        return float("inf")
    accountant = RDPAccountant()
    orders = np.arange(2, 64, 0.5)
    accountant.reset()
    accountant.compose_mechanism(
        "Gaussian", {"sigma": noise_multiplier, "sensitivity": 1}, steps, sample_rate
    )
    eps, _ = accountant.get_epsilon(delta, orders)
    return eps

print(f"Batch size: {batch_size}")
print(f"Total training steps: {total_training_steps}")
print(f"Delta: {delta}\n")

for sigma in noise_multipliers:
    print(f"Noise multiplier: {sigma}")
    epsilons = []
    for i, dataset_size in enumerate(client_dataset_sizes):
        sample_rate = batch_size / dataset_size
        eps = get_epsilon(sample_rate, sigma, total_training_steps, delta)
        epsilons.append(eps)
        print(f"  Client {i} - Dataset size: {dataset_size}, Sample rate: {sample_rate:.4f}, ε: {eps:.2f}")
    max_eps = max(epsilons)
    print(f"  Max ε across clients: {max_eps:.2f}\n")
