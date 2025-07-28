import unittest
from unittest.mock import MagicMock, patch
import torch
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from server.grpc_dp_server import SplitLearningService

class TestSplitLearningServer(unittest.TestCase):
    def setUp(self):
        self.server = SplitLearningService()
        self.server.dp_accountant = None  # Reset accountant each test

        self.batch_size = 10
        self.seq_len = 32
        self.hidden_dim = 768

        # Create fake activation tensor with gradients enabled
        self.activations = torch.randn((self.batch_size, self.seq_len, self.hidden_dim), requires_grad=True).to(self.server.device)
        self.attention_mask = torch.ones((self.batch_size, self.seq_len), dtype=torch.float32).to(self.server.device)
        self.labels = torch.randint(0, 2, (self.batch_size,), dtype=torch.long).to(self.server.device)

        self.dataset_size = 100
        self.delta = 1.0 / self.dataset_size

    def test_dp_workflow(self):
        print("Running test_dp_workflow...", flush=True)

        # Forward and backward pass
        outputs = self.server.model(self.activations, self.attention_mask)
        loss = self.server.criterion(outputs, self.labels)
        loss.backward()

        raw_grad = self.activations.grad.detach()
        total_norm = torch.norm(raw_grad).item()

        clipped_grad = self.server.clip_threshold and torch.nn.utils.clip_grad_norm_(
            [self.activations], self.server.clip_threshold
        )
        clipped_grad = raw_grad if clipped_grad is None else raw_grad  # fallback

        # Initialize DP accountant
        self.server.dp_accountant = self.server.dp_accountant or self._init_accountant()

        noisy_grad = clipped_grad + self.server.dp_accountant.noise_multiplier * torch.randn_like(clipped_grad)
        self.server.dp_accountant.step()

        epsilon = self.server.dp_accountant.get_epsilon(self.delta)
        
        print(f"DP epsilon after step: {epsilon}", flush=True)

        self.assertTrue(epsilon > 0)
        self.assertEqual(raw_grad.shape, noisy_grad.shape)

    def _init_accountant(self):
        from dp_utils.dp_utils import DPAccountant
        return DPAccountant(
            noise_multiplier=1.0,
            sample_rate=self.batch_size / self.dataset_size,
            steps=0
        )

    @patch("builtins.open")
    @patch("csv.writer")
    def test_logging_called(self, mock_csv_writer, mock_open):
        print("Running test_logging_called...", flush=True)

        # Simulate a gRPC context
        context = MagicMock()
        context.invocation_metadata.return_value = [
            ('client-id', '0'),
            ('dataset-size', str(self.dataset_size))
        ]

        # Create fake request
        act_np = self.activations.detach().cpu().numpy().flatten().astype(np.float32)
        mask_np = self.attention_mask.detach().cpu().numpy().flatten().astype(np.float32)
        labels = self.labels.detach().cpu().numpy().tolist()

        from grpc_generated import split_pb2

        request = split_pb2.ActivationRequest(
            activations=act_np,
            shape=[self.batch_size, self.seq_len, self.hidden_dim],
            attention_mask=mask_np,
            mask_shape=[self.batch_size, self.seq_len],
            labels=labels
        )

        response = self.server.SendActivations(request, context)
        
        print(f"Response loss: {response.loss}, accuracy: {response.accuracy}", flush=True)

        self.assertIsNotNone(response)
        self.assertTrue(response.loss > 0)
        self.assertTrue(response.accuracy >= 0)
        self.assertEqual(len(response.gradients), self.batch_size * self.seq_len * self.hidden_dim)

if __name__ == "__main__":
    unittest.main(verbosity=2)
