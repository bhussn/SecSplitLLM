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
    reconstruct_weights  # NEW: Add this import
)
from typing import Dict
import time
import os
from ..common.csv_logger import log_to_csv
import numpy as np
import torch.distributed as dist
from datetime import timedelta

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
        self.smpc_initialized = False  # Track SMPC initialization status

def initialize_smpc(self, smpc_world_size: int):
    if not self.smpc_initialized:
        rank = self.cid + 1
        world_size = smpc_world_size
        
        # Use unique port per client
        port = 29500  # Same port for every client
        
        # Set environment variables
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = str(port)  # Unique port
        os.environ['RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        
        # Add connection retry logic
        for attempt in range(3):
            try:
                dist.init_process_group(
                    backend="gloo",
                    init_method="env://",
                    world_size=world_size,
                    rank=rank,
                    timeout=timedelta(seconds=10))  # Timeout
                break
            except RuntimeError as e:
                if "Address already in use" in str(e) and attempt < 2:
                    wait_time = 2 ** attempt
                    print(f"Retry {attempt+1}/3 in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise
        
        crypten.init()
        self.smpc_initialized = True
 
def get_parameters(self, config: Config) -> Parameters:
        """Return the current local LoRA parameters."""
        print(f"[Client {self.cid}] get_parameters")
        lora_weights = get_lora_weights(self.net)
        return ndarrays_to_parameters(lora_weights)

def fit(self, ins: FitIns) -> FitRes:
        start_time = time.time()
        config = ins.config or {}
        smpc_world_size = config.get("smpc_world_size", 2)
        round_num = config.get("round", 0)
        
        # Initialize SMPC environment
        self.initialize_smpc(smpc_world_size)
        
        # --- DECRYPTION PHASE: Only needed after first round ---
        if round_num > 1:
            print(f"[Client {self.cid}] Decrypting aggregated model for round {round_num}")
            
            # Convert server parameters to byte string
            serialized_encrypted = b"".join(
                [arr.tobytes() for arr in parameters_to_ndarrays(ins.parameters)]
            )
            
            # Decrypt in SMPC context
            with crypten.mpc.ctx(rank=self.cid, world_size=smpc_world_size):
                encrypted_aggregate = crypten.serial.load(serialized_encrypted)
                plaintext_weights = reconstruct_weights(encrypted_aggregate)
            
            # Update local model with decrypted weights
            set_lora_weights(self.net, plaintext_weights)
        else:
            # First round - use plaintext parameters directly
            set_lora_weights(self.net, parameters_to_ndarrays(ins.parameters))
        
        # --- LOCAL TRAINING PHASE ---
        train_loss, train_acc = train_and_metrics(
            self.net, 
            self.trainloader, 
            epochs=int(config.get("local_epochs", 1))
        )
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

        # --- ENCRYPTION PHASE ---
        print(f"[Client {self.cid}] Encrypting updates for round {round_num}")
        with crypten.mpc.ctx(rank=self.cid, world_size=smpc_world_size):
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

def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """Evaluate the global model on the local test set."""
        parameters = ins.parameters
        
        # Handle encrypted models for evaluation
        if ins.config.get("round", 0) > 1:
            smpc_world_size = ins.config.get("smpc_world_size", 3)
            self.initialize_smpc(smpc_world_size)
            
            serialized_encrypted = b"".join(
                [arr.tobytes() for arr in parameters_to_ndarrays(parameters)]
            )
            
            with crypten.mpc.ctx(rank=self.cid, world_size=smpc_world_size):
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


def client_fn(context: Context) -> fl.client.Client:
    node_id = int(context.node_id)
    print(f"--- Creating client with node_id: {node_id} ---")
    return FlowerSmpcClient(cid=node_id)

app = fl.client.ClientApp(client_fn=client_fn)