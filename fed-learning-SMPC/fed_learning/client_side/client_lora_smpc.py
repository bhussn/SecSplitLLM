import flwr as fl
import warnings
import crypten
from flwr.common import (
    Config,
    Context,
    Parameters,
    FitIns,      # Import FitIns
    FitRes,
    EvaluateIns, # Import EvaluateIns
    EvaluateRes,
    Scalar,
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
)
from typing import Dict, Tuple
from fed_learning.smpc.smpc_aggregator import secure_aggregate
import csv
from datetime import datetime
import time
import os
from ..common.csv_logger import log_to_csv
import io
import numpy as np

# Suppress warnings from CrypTen
warnings.filterwarnings("ignore", category=UserWarning)

class FlowerSmpcClient(fl.client.Client):
    """Flower client for LoRA fine-tuning with SMPC."""

    def __init__(self, cid: int):
        self.cid = cid
        self.csv_path = f"fed_learning/results/client_{cid}_metrics.csv"
        self.training_metrics = []  # Store per-epoch metrics

        self.net = load_lora_model()
        self.trainloader, self.testloader = load_data(self.cid, num_partitions=3)

    def get_parameters(self, config: Config) -> Parameters:
        """Return the current local LoRA parameters."""
        print(f"[Client {self.cid}] get_parameters")
        lora_weights = get_lora_weights(self.net)
        return ndarrays_to_parameters(lora_weights)

    def fit(self, ins: FitIns) -> FitRes:
        parameters = ins.parameters
        config = ins.config
        start_time = time.time()
    
        # Set model weights
        set_lora_weights(self.net, parameters_to_ndarrays(parameters))
    
        # Train locally and collect metrics in a single pass
        train_loss, train_acc = train_and_metrics(  # NEW FUNCTION
        self.net, 
        self.trainloader, 
        epochs=int(config["local_epochs"])
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
                datetime.now().isoformat(),
                config["round"],
                train_loss,
                train_acc,
                update_size,
                comp_time
            ]
        )

        print(f"[Client {self.cid}] encrypting {len(updated_weights)} LoRA layers...")
        encrypted_weights = secret_share_weights(updated_weights)
        print(f"[Client {self.cid}] encryption complete.")

        # Serialize Cryptensors to bytes
        serialized_weights = []
        for tensor in encrypted_weights:
            with io.BytesIO() as f:
                crypten.save(tensor, f) 
                f.seek(0)
                serialized_weights.append(np.frombuffer(f.getvalue(), dtype=np.uint8))
    
        # Calculate communication cost (size of encrypted weights)
        comm_cost = 0
        for tensor in encrypted_weights:
            if hasattr(tensor, "share") and hasattr(tensor.share, "nelement"):
                # Debug mode - use share size
                comm_cost += tensor.share.nelement() * tensor.share.element_size()
            else:
                # Production mode - use tensor size as fallback
                comm_cost += tensor.nelement() * tensor.element_size()
    
        return FitRes(
        status=fl.common.Status(code=fl.common.Code.OK, message="Success"),
        parameters=ndarrays_to_parameters(serialized_weights),
        num_examples=len(self.trainloader.dataset),
        metrics={"comm_cost_bytes": comm_cost},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """Evaluate the global model on the local test set."""
        # Unpack parameters from the instruction object
        parameters = ins.parameters
        
        print(f"[Client {self.cid}] starting evaluation...")
        set_lora_weights(self.net, parameters_to_ndarrays(parameters))
        loss, results = test(self.net, self.testloader)
        print(f"[Client {self.cid}] evaluation finished. Loss: {loss:.4f}")
        
        return EvaluateRes(
            status=fl.common.Status(code=fl.common.Code.OK, message="Success"),
            loss=float(loss),
            num_examples=len(self.testloader.dataset),
            metrics=results,
        )


def client_fn(context: Context) -> fl.client.Client:
    """Create a Flower client instance for simulation."""
    node_id = int(context.node_id)
    print(f"--- Creating client with node_id: {node_id} ---")
    return FlowerSmpcClient(cid=node_id)

# The Flower App for the client
app = fl.client.ClientApp(client_fn=client_fn)
