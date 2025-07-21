import flwr as fl
import warnings
import crypten
from flwr.common import (
    Config,
    Context,
    Parameters,
    FitIns,
    FitRes,
    EvaluateIns,
    EvaluateRes,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from fed_learning.training.smpc_trainer import (
    load_lora_model,
    load_data,
    train_and_metrics,
    test,
    get_lora_weights,
    set_lora_weights,
    secret_share_weights,
    reconstruct_weights
)
from typing import Dict
import time
import os
from ..common.csv_logger import log_to_csv
import numpy as np
import torch.distributed as dist
from datetime import timedelta
import traceback

# Suppress warnings from CrypTen
warnings.filterwarnings("ignore", category=UserWarning)

class FlowerSmpcClient(fl.client.Client):
    """Flower client for LoRA fine-tuning with SMPC."""

    def __init__(self, cid: int):
        self.cid = cid
        self.csv_path = f"fed_learning/results/client_{cid}_metrics.csv"
        self.smpc_initialized = False
        
        # Create results directory if not exists
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        
        # Initialize model and data
        self.net = load_lora_model()
        self.trainloader, self.testloader = load_data(self.cid, num_partitions=10)

    def initialize_smpc(self, smpc_world_size: int):
        if not self.smpc_initialized:
            rank = self.cid + 1
            world_size = smpc_world_size
        
            print(f"[Client {self.cid}] Initializing SMPC (rank {rank}/{world_size})")
            
            # Add robust connection retry logic with backoff
            max_attempts = 20  # Increased attempts
            for attempt in range(max_attempts):
                try:
                    dist.init_process_group(
                        backend="gloo",
                        init_method="env://", # Use existing enviromental variables
                        world_size=world_size,
                        rank=rank,
                        timeout=timedelta(minutes=30)  # Longer timeout
                    )
                    print(f"[Client {self.cid}] Distributed process group initialized")
                    break
                except RuntimeError as e:
                    if attempt < max_attempts - 1:
                        wait_time = 15 * (attempt + 1)  # Longer backoff
                        print(f"[Client {self.cid}] SMPC init failed (attempt {attempt+1}/{max_attempts}): {str(e)}")
                        print(f"[Client {self.cid}] Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        error_msg = f"SMPC initialization failed after {max_attempts} attempts"
                        print(f"[Client {self.cid}] {error_msg}")
                        raise RuntimeError(error_msg)
            
            # Initialize CrypTen after process group
            crypten.init()
            self.smpc_initialized = True
            print(f"[Client {self.cid}] CrypTen initialized")
            
            # Create readiness file
            open(f"/tmp/client_{self.cid}.ready", "w").close()
            print(f"[Client {self.cid}] Readiness file created")

    def get_parameters(self, config: Config) -> Parameters:
        """Return the current local LoRA parameters."""
        if not self.smpc_initialized:
            # Ensure SMPC is initialized before any operation
            smpc_world_size = config.get("smpc_world_size")
            self.initialize_smpc(smpc_world_size)

        print(f"[Client {self.cid}] get_parameters")
        lora_weights = get_lora_weights(self.net)
        return ndarrays_to_parameters(lora_weights)

    def fit(self, ins: FitIns) -> FitRes:
        start_time = time.time()
        config = ins.config or {}
        smpc_world_size = config.get("smpc_world_size", 4)  # Default to 4
        round_num = config.get("round", 0)
        
        # Initialize SMPC environment with error handling
        try:
            self.initialize_smpc(smpc_world_size)
        except Exception as e:
            
            error_msg = f"SMPC initialization failed: {str(e)}"
            print(f"[Client {self.cid}] {error_msg}")
            traceback.print_exc()
            return FitRes(
                status=fl.common.Status(code=fl.common.Code.FIT_ERROR, message=error_msg),
                parameters=ins.parameters,
                num_examples=0,
                metrics={}
            )
        
        # --- DECRYPTION PHASE ---
        try:
            if round_num > 1:
                print(f"[Client {self.cid}] Decrypting aggregated model for round {round_num}")
                
                # Convert server parameters to byte string
                with crypten.mpc.ctx(rank=self.cid + 1, world_size=smpc_world_size):
                    serialized_encrypted = b"".join(
                    [arr.tobytes() for arr in parameters_to_ndarrays(ins.parameters)]
                    )
                
                # Decrypt in SMPC context with proper rank (self.cid + 1)
                encrypted_aggregate = crypten.serial.load(serialized_encrypted)
                plaintext_weights = reconstruct_weights(encrypted_aggregate)
                
                # Update local model with decrypted weights
                set_lora_weights(self.net, plaintext_weights)
            else:
                # First round - use plaintext parameters directly
                set_lora_weights(self.net, parameters_to_ndarrays(ins.parameters))
        except Exception as e:
            error_msg = f"Decryption failed: {str(e)}"
            print(f"[Client {self.cid}] {error_msg}")
            traceback.print_exc()
            return FitRes(
                status=fl.common.Status(code=fl.common.Code.FIT_ERROR, message=error_msg),
                parameters=ins.parameters,
                num_examples=0,
                metrics={}
            )
        
        # --- LOCAL TRAINING PHASE ---
        try:
            train_loss, train_acc = train_and_metrics(
                self.net, 
                self.trainloader, 
                epochs=int(config.get("local_epochs", 1)))
            comp_time = time.time() - start_time

            # Get update size
            updated_weights = get_lora_weights(self.net)
            update_size = sum(p.nbytes for p in updated_weights)

            # Log to CSV
            log_to_csv(
                file_path=self.csv_path,
                headers=["timestamp", "round", "train_loss", "train_acc", "update_size_bytes", "comp_time_sec"],
                row_data=[
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    round_num,
                    train_loss,
                    train_acc,
                    update_size,
                    comp_time
                ]
            )
        except Exception as e:
            comp_time = time.time() - start_time  # Calculate anyway
            error_msg = f"Training failed: {str(e)}"
            print(f"[Client {self.cid}] {error_msg}")
            traceback.print_exc()
            return FitRes(
                status=fl.common.Status(code=fl.common.Code.FIT_ERROR, message=error_msg),
                parameters=ins.parameters,
                num_examples=0,
                metrics={}
            )

        # --- ENCRYPTION PHASE ---
        try:
            print(f"[Client {self.cid}] Encrypting updates for round {round_num}")
            # Use proper rank in SMPC context (self.cid + 1)
            with crypten.mpc.ctx(rank=self.cid+1, world_size=smpc_world_size):
                encrypted_weights = secret_share_weights(updated_weights)
                serialized = crypten.serial.serialize(encrypted_weights)
            
            comm_cost = len(serialized)
            print(f"[Client {self.cid}] Encryption complete. Size: {comm_cost} bytes")

            return FitRes(
                status=fl.common.Status(code=fl.common.Code.OK, message="Success"),
                parameters=ndarrays_to_parameters(
                    [np.frombuffer(serialized, dtype=np.uint8)]
                ),
                num_examples=len(self.trainloader.dataset),
                metrics={"comm_cost_bytes": comm_cost},
            )
        except Exception as e:
            error_msg = f"Encryption failed: {str(e)}"
            print(f"[Client {self.cid}] {error_msg}")
            traceback.print_exc()
            return FitRes(
                status=fl.common.Status(code=fl.common.Code.FIT_ERROR, message=error_msg),
                parameters=ins.parameters,
                num_examples=len(self.trainloader.dataset),
                metrics={}
            )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """Evaluate the global model on the local test set."""
        try:
            parameters = ins.parameters
            round_num = ins.config.get("round", 0)
            smpc_world_size = ins.config.get("smpc_world_size", 4)
            
            # Initialize SMPC environment
            self.initialize_smpc(smpc_world_size)
            
            # Handle encrypted models for evaluation
            if round_num > 1:
                print(f"[Client {self.cid}] Decrypting model for evaluation in round {round_num}")
                
                serialized_encrypted = b"".join(
                    [arr.tobytes() for arr in parameters_to_ndarrays(parameters)]
                )
                
                # Use proper rank in SMPC context (self.cid + 1)
                with crypten.mpc.ctx(rank=self.cid+1, world_size=smpc_world_size):
                    encrypted_aggregate = crypten.serial.load(serialized_encrypted)
                    plaintext_weights = reconstruct_weights(encrypted_aggregate)
                
                set_lora_weights(self.net, plaintext_weights)
            else:
                set_lora_weights(self.net, parameters_to_ndarrays(parameters))
            
            loss, results = test(self.net, self.testloader)
            print(f"[Client {self.cid}] Evaluation finished. Loss: {loss:.4f}, Accuracy: {results['accuracy']:.4f}")
            
            return EvaluateRes(
                status=fl.common.Status(code=fl.common.Code.OK, message="Success"),
                loss=float(loss),
                num_examples=len(self.testloader.dataset),
                metrics=results,
            )
        except Exception as e:
            error_msg = f"Evaluation failed: {str(e)}"
            print(f"[Client {self.cid}] {error_msg}")
            traceback.print_exc()
            return EvaluateRes(
                status=fl.common.Status(code=fl.common.Code.EVALUATE_ERROR, message=error_msg),
                loss=0.0,
                num_examples=0,
                metrics={},
            )


def client_fn(context: Context) -> fl.client.Client:
    node_id = int(context.node_id)
    print(f"--- Creating client with node_id: {node_id} ---")
    return FlowerSmpcClient(cid=node_id)

app = fl.client.ClientApp(client_fn=client_fn)

def main(node_id: int):
    print(f"[Client {node_id}] Starting client")
    client = FlowerSmpcClient(cid=node_id)
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8082",
        client=client,
    )

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python client_lora_smpc.py <node_id>")
        sys.exit(1)
    main(int(sys.argv[1]))
