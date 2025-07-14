import crypten
import numpy as np
from typing import List

def secure_aggregate(encrypted_updates, num_examples_list):
    """Perform actual secure aggregation"""
    total_examples = sum(num_examples_list)
    aggregated = []
    
    # Process each layer separately
    for layer_idx in range(len(encrypted_updates[0])):
        layer_aggregate = None
        
        # Weighted aggregation
        for client_idx, client_shares in enumerate(encrypted_updates):
            weighted_tensor = client_shares[layer_idx] * num_examples_list[client_idx]
            
            if layer_aggregate is None:
                layer_aggregate = weighted_tensor
            else:
                layer_aggregate += weighted_tensor
        
        # Average and store
        aggregated.append(layer_aggregate / total_examples)
    
    return aggregated