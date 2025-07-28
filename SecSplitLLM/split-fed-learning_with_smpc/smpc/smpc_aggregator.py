import crypten
import torch
import numpy as np
from crypten.mpc import run_multiprocess

# Optional: import PySyft for Shamir secret sharing
try:
    import syft as sy
    from syft.frameworks.torch.fl import utils as syft_utils
    syft_available = True
except ImportError:
    syft_available = False

def encrypt_client_update(update, scheme="additive", workers=None):
    """Encrypt a list of NumPy arrays (model layers) using the specified scheme."""
    if scheme == "additive":
        return [crypten.cryptensor(torch.tensor(layer)) for layer in update]
    elif scheme == "shamir":
        if not syft_available:
            raise ImportError("PySyft is required for Shamir secret sharing.")
        if workers is None:
            raise ValueError("Workers must be provided for Shamir secret sharing.")
        return [torch.tensor(layer).share(*workers, crypto_provider=workers[0]) for layer in update]
    else:
        raise ValueError(f"Unsupported scheme: {scheme}")

def aggregate_encrypted_updates(encrypted_updates, scheme="additive"):
    """Aggregate encrypted updates layer-wise."""
    num_layers = len(encrypted_updates[0])
    aggregated = []
    for i in range(num_layers):
        layer_sum = encrypted_updates[0][i]
        for j in range(1, len(encrypted_updates)):
            layer_sum += encrypted_updates[j][i]
        aggregated.append(layer_sum / len(encrypted_updates))
    return aggregated

def decrypt_aggregated_update(aggregated_encrypted, scheme="additive"):
    """Decrypt the aggregated encrypted update."""
    if scheme == "additive":
        return [layer.get_plain_text().numpy() for layer in aggregated_encrypted]
    elif scheme == "shamir":
        return [layer.get().numpy() for layer in aggregated_encrypted]
    else:
        raise ValueError(f"Unsupported scheme: {scheme}")

def secure_aggregation(client_updates, scheme="additive", world_size=3):
    """
    Perform secure aggregation on client updates using the specified scheme.    
    Each client update is a list of NumPy arrays (model layers).
    """
    @run_multiprocess(world_size=world_size)
    def _secure_aggregation():
        crypten.init()
        rank = crypten.communicator.get().get_rank()

        # Setup for Shamir
        workers = None
        if scheme == "shamir" and rank == 0:
            hook = sy.TorchHook(torch)
            workers = [sy.VirtualWorker(hook, id=f"worker{i}") for i in range(world_size)]

        # Each party encrypts its own update
        encrypted_updates = [encrypt_client_update(update, scheme, workers) for update in client_updates]

        # Only rank 0 performs aggregation and decryption
        if rank == 0:
            aggregated_encrypted = aggregate_encrypted_updates(encrypted_updates, scheme)
            aggregated_plain = decrypt_aggregated_update(aggregated_encrypted, scheme)
            return aggregated_plain
        
    return _secure_aggregation()









