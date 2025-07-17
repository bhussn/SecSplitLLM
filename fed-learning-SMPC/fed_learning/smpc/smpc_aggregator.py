import crypten
from typing import List
import time

def secure_aggregate(encrypted_updates: List, weights: List) -> List:
    """
    Perform secure weighted average aggregation of encrypted model updates.
    
    Args:
        encrypted_updates: List of client updates, each being a list of CrypTen tensors
        weights: List of weights (typically number of examples) for each client
        
    Returns:
        List of aggregated CrypTen tensors (still encrypted)
    """
    # 1. Input validation
    if not encrypted_updates:
        raise ValueError("No encrypted updates provided for aggregation")
    
    if len(encrypted_updates) != len(weights):
        raise ValueError("Mismatch between number of updates and weights")
    
    total_weight = sum(weights)
    if total_weight <= 0:
        raise ValueError("Total weight must be positive")
    
    aggregated = []
    num_layers = len(encrypted_updates[0])
    layer_times = []  # For performance monitoring
    
    # 2. Vectorized aggregation per layer
    for layer_idx in range(num_layers):
        layer_start = time.time()
        
        # Stack all client updates for this layer
        client_updates = [update[layer_idx] for update in encrypted_updates]
        
        # Convert weights to crypten tensor
        weights_tensor = crypten.tensor(weights)
        
        # Compute weighted sum (vectorized)
        stacked_updates = crypten.stack(client_updates)
        weighted_sum = stacked_updates.mul(weights_tensor.unsqueeze(1)).sum(dim=0)
        
        # Compute weighted average
        layer_agg = weighted_sum.div(total_weight)
        aggregated.append(layer_agg)
        
        layer_times.append(time.time() - layer_start)
    
    # 3. Log aggregation performance
    avg_time = sum(layer_times) / num_layers
    max_time = max(layer_times)
    print(f"[Aggregator] Processed {num_layers} layers | "
          f"Avg: {avg_time:.4f}s/layer | Max: {max_time:.4f}s")
    
    return aggregated