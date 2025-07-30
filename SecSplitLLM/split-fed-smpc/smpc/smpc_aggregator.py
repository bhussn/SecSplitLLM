import os
import io
import numpy as np
import torch
import crypten
from crypten.mpc.ptype import ptype

os.environ["CRYPTEN_USE_CPU"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

SAFE_GLOBALS = [
    crypten.mpc.mpc.MPCTensor,
    crypten.gradients.AutogradContext,
    crypten.mpc.primitives.arithmetic.ArithmeticSharedTensor,
    crypten.encoder.FixedPointEncoder,
    ptype,
]

# Result holder shared by rank 0 only
result_holder = {}


def encrypt_client_update(update, scheme="additive", workers=None):
    if scheme == "additive":
        encrypted_buffers = []
        for layer in update:
            enc_tensor = crypten.cryptensor(torch.tensor(layer))
            buffer = io.BytesIO()
            crypten.save(enc_tensor, buffer)
            encrypted_buffers.append(np.frombuffer(buffer.getvalue(), dtype=np.uint8))
        return encrypted_buffers
    else:
        raise ValueError(f"Unsupported scheme: {scheme}")


def deserialize_encrypted_update(serialized_update):
    deserialized = []
    for layer_bytes in serialized_update:
        buf = io.BytesIO(layer_bytes.tobytes())
        with torch.serialization.safe_globals(SAFE_GLOBALS):
            tensor = crypten.load(buf)
        deserialized.append(tensor)
    return deserialized


def _secure_aggregation_worker(client_updates_serialized, world_size):
    from crypten.communicator import get

    crypten.init()
    rank = get().get_rank()
    print(f"[CrypTen Rank {rank}] torch.cuda.is_available(): {torch.cuda.is_available()}")

    client_tensors = [deserialize_encrypted_update(update) for update in client_updates_serialized]
    num_layers = len(client_tensors[0])
    aggregated_layers = []

    for i in range(num_layers):
        agg = client_tensors[0][i]
        for j in range(1, len(client_tensors)):
            agg += client_tensors[j][i]
        aggregated_layers.append(agg / len(client_tensors))

    if rank == 0:
        decrypted = [layer.get_plain_text().cpu().numpy() for layer in aggregated_layers]
        result_holder["aggregated_flat"] = np.concatenate([w.flatten() for w in decrypted])


def aggregate_encrypted_updates(client_updates_serialized, world_size=3):
    """
    Runs secure aggregation on serialized client encrypted updates.
    Returns the aggregated flattened numpy array on rank 0,
    None on other ranks.
    """
    crypten.mpc.run_multiprocess(world_size=world_size)(_secure_aggregation_worker)(
        client_updates_serialized, world_size
    )
    return result_holder.get("aggregated_flat")


def split_weights(flat_weights, param_shapes):
    split_arrays = []
    pointer = 0
    for shape in param_shapes:
        size = np.prod(shape)
        split_arrays.append(flat_weights[pointer:pointer + size].reshape(shape))
        pointer += size
    return split_arrays
