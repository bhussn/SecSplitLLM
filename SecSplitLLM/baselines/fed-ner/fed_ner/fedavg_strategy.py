# fed_ner/fedavg_strategy.py

import logging
from flwr.server.strategy import FedAvg
from fed_ner.metrics import parameters_to_ndarrays

logger = logging.getLogger(__name__)

class CustomFedAvg(FedAvg):
    """
    Custom FedAvg strategy to update global model parameters.
    No client-side metric aggregation. Logging happens via centralized evaluation.
    """

    def __init__(self, initial_parameters, *args, **kwargs):
        super().__init__(initial_parameters=initial_parameters, *args, **kwargs)
        self.current_round = 0
        self._prev_params = parameters_to_ndarrays(initial_parameters)

    def aggregate_fit(self, rnd, results, failures):
        """Aggregate model weights from clients and update initial weights."""
        agg = super().aggregate_fit(rnd, results, failures)
        if agg:
            self.initial_parameters = agg[0]
            self._prev_params = parameters_to_ndarrays(agg[0])
        self.current_round += 1
        return agg

    def aggregate_evaluate(self, rnd, results, failures):
        """Pass through — no aggregation; centralized evaluation is used."""
        return super().aggregate_evaluate(rnd, results, failures)
