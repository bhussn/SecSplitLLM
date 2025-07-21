import flwr as fl
import os
import time
import csv
import numpy as np
import crypten
from datetime import datetime
from typing import List, Tuple, Dict, Optional, Union
from flwr.common import (
    Metrics,
    Parameters,
    FitRes,
    EvaluateRes,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    Scalar
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.server import ServerApp
from flwr.server.server import ServerConfig
from fed_learning.training.smpc_trainer import load_lora_model, get_lora_weights
from fed_learning.smpc.smpc_aggregator import secure_aggregate
import torch.distributed as dist
from datetime import timedelta
import traceback

class FedAvgSecure(FedAvg):
    def __init__(self, smpc_world_size=None, *args, **kwargs):
        # Get world size from environment if not provided
        if smpc_world_size is None:
            smpc_world_size = int(os.environ.get('WORLD_SIZE', 4))
        super().__init__(*args, **kwargs)
        self.smpc_world_size = smpc_world_size
        self.csv_path = "fed_learning/results/server_metrics.csv"
        # Create results directory if not exists
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        self.round_start_time = None
        self.client_comm_costs = {}
        self.crypten_initialized = False
        print("[Server] SMPC will be initialized during first aggregation")
    
    def configure_fit(self, server_round, parameters, client_manager):
        self.round_start_time = time.time()
        self.client_comm_costs = {}

        if server_round == 1 and not self.crypten_initialized:
            print("[Server] Initializing CrypTen before instructing clients...")
            self.initialize_crypten()
            
        return super().configure_fit(server_round, parameters, client_manager)
    
    def initialize_crypten(self):
        if not self.crypten_initialized:
            # Only set if not already set
            if 'MASTER_ADDR' not in os.environ:
                os.environ['MASTER_ADDR'] = '127.0.0.1'
            if 'MASTER_PORT' not in os.environ:
                os.environ['MASTER_PORT'] = '29500'
            os.environ['RANK'] = '0'  # Server is always rank 0
            os.environ['WORLD_SIZE'] = str(self.smpc_world_size)
            
            print(f"[Server] Initializing SMPC (rank=0, world_size={self.smpc_world_size})")
            
            # Initialize distributed system with retries
            max_attempts = 10  # Increased attempts
            for attempt in range(max_attempts):
                try:
                    dist.init_process_group(
                        backend="gloo",
                        init_method="env://",
                        world_size=self.smpc_world_size,
                        rank=0,
                        timeout=timedelta(minutes=30),  # Increased timeout
                    )
                    print("[Server] Distributed process group initialized")
                    break
                except RuntimeError as e:
                    if attempt < max_attempts - 1:
                        wait_time = 15 * (attempt + 1)  # Exponential backoff
                        print(f"[Server] SMPC init failed (attempt {attempt+1}/{max_attempts}): {str(e)}")
                        print(f"[Server] Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        error_msg = f"SMPC initialization failed after {max_attempts} attempts"
                        print(f"[Server] {error_msg}")
                        raise RuntimeError(error_msg)
            
            # Then initialize CrypTen
        if not self.crypten_initialized:
            crypten.init()
            self.crypten_initialized = True
            print("[Server] CrypTen initialized")
            
            # Create readiness file
            open("/tmp/server.ready", "w").close()
            print("[Server] Readiness file created")
        else:
            print("[Server] SMPC already initialized")

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}
        
        if not self.crypten_initialized:
            self.initialize_crypten()
        
        # 1. Record communication costs
        for client, fit_res in results:
            cid = client.cid
            self.client_comm_costs[cid] = fit_res.metrics.get("comm_cost_bytes", 0)
        
        # 2. Time the secure aggregation process
        agg_start_time = time.time()
        
        try:
            print(f"[Round {server_round}] Processing {len(results)} client updates")
            encrypted_updates = []
            num_examples_list = []
            
            # Use SMPC context with server as rank 0
            with crypten.mpc.ctx(rank=0, world_size=self.smpc_world_size):
                # 4. Deserialize client updates
                for _, fit_res in results:
                    # Client sends a single serialized bytestring
                    serialized_data = b"".join(
                        [arr.tobytes() for arr in parameters_to_ndarrays(fit_res.parameters)]
                    )
                    client_tensors = crypten.serial.load(serialized_data)
                    encrypted_updates.append(client_tensors)
                    num_examples_list.append(fit_res.num_examples)
                
                # 5. Perform secure aggregation
                print(f"[Round {server_round}] Performing secure aggregation")
                aggregated_ndarrays = secure_aggregate(encrypted_updates, num_examples_list)
                
                # 6. Serialize aggregated result
                serialized_agg = crypten.serial.serialize(aggregated_ndarrays)
            
            # 7. Prepare parameters to send back to clients
            agg_parameters = ndarrays_to_parameters(
                [np.frombuffer(serialized_agg, dtype=np.uint8)]
            )
        
        except Exception as e:
            print(f"[Server] Aggregation failed: {str(e)}")
            traceback.print_exc()
            # Fallback to standard FedAvg
            print("[Server] Falling back to standard FedAvg aggregation")
            return super().aggregate_fit(server_round, results, failures)
        
        # 8. Calculate metrics
        agg_latency = time.time() - agg_start_time
        total_comm_bytes = sum(self.client_comm_costs.values())
        round_time = time.time() - self.round_start_time
        
        # 9. Log to CSV
        try:
            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    server_round,
                    agg_latency,
                    total_comm_bytes,
                    len(results),
                    round_time
                ])
        except Exception as e:
            print(f"[Server] Failed to log metrics: {str(e)}")
        
        # 10. Return serialized encrypted aggregate
        return agg_parameters, {}

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    total_examples = sum(num_examples for num_examples, _ in metrics)
    aggregated_metrics = {}
    if total_examples == 0:
        return aggregated_metrics
        
    for metric_name in ["precision", "recall", "f1", "accuracy"]:
        weighted_sum = sum(m.get(metric_name, 0) * num_examples for num_examples, m in metrics)
        aggregated_metrics[metric_name] = weighted_sum / total_examples
    return aggregated_metrics

def fit_config(server_round: int) -> Dict:
    # Get world size from environment with default 4
    world_size = int(os.environ.get('WORLD_SIZE', 4))
    return {
        "round": server_round,
        "local_epochs": 1,
        "smpc_world_size": world_size,  # Consistent with environment
        "num_partitions": 10
    }

# Main method
def main():
    # Initialize model
    print("[Server] Loading initial model...")
    initial_model = load_lora_model()
    initial_parameters = ndarrays_to_parameters(get_lora_weights(initial_model))
    
    # Create strategy
    print("[Server] Creating strategy...")
    try:
        world_size = int(os.environ.get('WORLD_SIZE', 2))
        strategy = FedAvgSecure(
            smpc_world_size=world_size,
            fraction_fit=1.0,
            fraction_evaluate=0.5,
            min_fit_clients=world_size-1,
            min_evaluate_clients=world_size-1,
            min_available_clients=world_size-1,
            initial_parameters=initial_parameters,
            evaluate_metrics_aggregation_fn=weighted_average,
            on_fit_config_fn=fit_config,
        )
    except Exception as e:
        print(f"[Server] Strategy creation failed: {str(e)}")
        traceback.print_exc()
        return
    
    # Server configuration
    config = ServerConfig(num_rounds=3)
    
    # Start server
    print("[Server] Starting Flower server on port 8082")
    fl.server.start_server(
        server_address="0.0.0.0:8082",
        config=config,
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
