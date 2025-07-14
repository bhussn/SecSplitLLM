import flwr as fl
import inspect
import pkgutil
from flwr.common import (
    Metrics,
    Parameters,
    FitRes,
    EvaluateRes,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    Scalar,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.server.server import ServerConfig
from flwr.server import ServerApp
from typing import List, Tuple, Dict, Optional, Union
import crypten
import torch
import io
import numpy as np
import time
import csv
from datetime import datetime
from fed_learning.training.smpc_trainer import load_lora_model, get_lora_weights
from fed_learning.smpc.smpc_aggregator import secure_aggregate
import importlib

# Get ALL classes from crypten and its submodules
# Recursive function to get all classes in crypten and its submodules
def get_all_crypten_classes():
    classes = []
    modules_to_check = [crypten]
    visited = set()
    
    while modules_to_check:
        mod = modules_to_check.pop()
        mod_name = mod.__name__
        if mod_name in visited:
            continue
        visited.add(mod_name)
        
        # Skip deprecated models module
        if mod_name == "crypten.models":
            print("Skipping deprecated crypten.models module")
            continue
            
        # Get classes from current module
        for name, obj in inspect.getmembers(mod):
            if inspect.isclass(obj) and obj.__module__.startswith('crypten'):
                classes.append(obj)
        
        # Get submodules
        if hasattr(mod, '__path__'):
            for _, name, is_pkg in pkgutil.iter_modules(mod.__path__):
                full_name = f"{mod_name}.{name}"
                
                # Skip deprecated models module
                if full_name == "crypten.models":
                    print("Skipping deprecated crypten.models module")
                    continue
                    
                if full_name in visited:
                    continue
                    
                try:
                    module = importlib.import_module(full_name)
                    modules_to_check.append(module)
                except DeprecationWarning as e:
                    print(f"Skipping deprecated module {full_name}: {e}")
                except Exception as e:
                    print(f"Skipping module {full_name} due to error: {e}")
    
    return list(set(classes))

# Get all CrypTen classes
crypten_classes = get_all_crypten_classes()
print(f"Found {len(crypten_classes)} CrypTen classes for safe deserialization")

# --- 1. Define a Secure Strategy for Aggregation ---
class FedAvgSecure(FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.csv_path = "fed_learning/results/server_metrics.csv"
        self.round_start_time = None
        self.client_comm_costs = {}
        
        # Initialize CrypTen once when server starts
        if not crypten.is_initialized():
            crypten.init()
            print("Crypten initialized on server start")
    
    def __del__(self):
        # Clean up when server is destroyed
        if crypten.is_initialized():
            crypten.uninit()
            print("Crypten uninitialized on server exit")
    
    def configure_fit(self, server_round, parameters, client_manager):
        self.round_start_time = time.time()
        self.client_comm_costs = {}
        return super().configure_fit(server_round, parameters, client_manager)
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}
        
        # 1. Record communication costs
        for client, fit_res in results:
            cid = client.cid
            self.client_comm_costs[cid] = fit_res.metrics.get("comm_cost_bytes", 0)
        
        # 2. Time the secure aggregation process
        agg_start_time = time.time()
        encrypted_updates = []
        num_examples_list = []
        
        try:
            for _, fit_res in results:
                client_serialized = parameters_to_ndarrays(fit_res.parameters)
                client_tensors = []
                
                for arr in client_serialized:
                    with io.BytesIO(arr.tobytes()) as f:
                        # Use comprehensive allowlist
                        with torch.serialization.safe_globals(crypten_classes):
                            client_tensors.append(crypten.load(f))
                
                encrypted_updates.append(client_tensors)
                num_examples_list.append(fit_res.num_examples)
            
            # 3. Perform secure aggregation
            print(f"[Round {server_round}] Performing secure aggregation with {len(encrypted_updates)} updates")
            aggregated_ndarrays = secure_aggregate(encrypted_updates, num_examples_list)
            
            # 4. DECRYPT FINAL RESULT ONLY
            plaintext_aggregate = [tensor.get_plain_text().numpy() for tensor in aggregated_ndarrays]
            
        except Exception as e:
            print(f"Aggregation failed: {e}")
            raise
        
        # 5. Calculate metrics
        agg_latency = time.time() - agg_start_time
        total_comm_bytes = sum(self.client_comm_costs.values())
        round_time = time.time() - self.round_start_time
        
        # 6. Log to CSV
        try:
            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(),
                    server_round,
                    agg_latency,
                    total_comm_bytes,
                    len(results),
                    round_time
                ])
        except Exception as e:
            print(f"Failed to log metrics: {e}")
        
        # 7. Return plaintext aggregate
        return ndarrays_to_parameters(plaintext_aggregate), {}


# --- 2. Metric Aggregation and Configuration ---
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate evaluation results using a weighted average."""
    total_examples = sum(num_examples for num_examples, _ in metrics)
    aggregated_metrics = {}
    if total_examples == 0:
        return aggregated_metrics
        
    for metric_name in ["precision", "recall", "f1", "accuracy"]:
        weighted_sum = sum(m.get(metric_name, 0) * num_examples for num_examples, m in metrics)
        aggregated_metrics[metric_name] = weighted_sum / total_examples
    return aggregated_metrics


def fit_config(server_round: int) -> Dict:
    """Return training configuration for each round."""
    return {
        "round": server_round,
        "local_epochs": 2, # As defined in pyproject.toml
    }


# --- 3. Server Initialization and App ---
# Initialize the model to get the initial LoRA parameters
initial_model = load_lora_model()
initial_parameters = ndarrays_to_parameters(get_lora_weights(initial_model))

# Create the secure strategy instance
strategy = FedAvgSecure(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=3,
    min_evaluate_clients=3,
    min_available_clients=3,
    initial_parameters=initial_parameters,
    evaluate_metrics_aggregation_fn=weighted_average,
    on_fit_config_fn=fit_config,
)

# Define the server configuration
config = ServerConfig(num_rounds=3)

# The Flower App for the server
app = ServerApp(
    config=config,
    strategy=strategy,
)
