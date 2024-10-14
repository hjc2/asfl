import numpy as np
from typing import Union, Dict, List, Optional, Tuple
from flwr.common import (
    FitRes,
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    Scalar,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate
from .fed_custom import FedCustom

class FedFuzz(FedCustom):
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        weights_results = []

        for _, fit_res in results:
            loss = fit_res.metrics['loss']
            accuracy = fit_res.metrics['accuracy']
            weight = self.calculate_fuzzy_weight(loss, accuracy)

            weights_results.append((parameters_to_ndarrays(fit_res.parameters), weight))

        return self.aggregate_weights(weights_results)

    def calculate_fuzzy_weight(self, loss: float, accuracy: float) -> float:
        # Fuzzy logic membership functions
        def loss_membership(loss_value: float) -> Tuple[float, float, float]:  # Low, Medium, High
            low = max(0, (1 - loss_value) / 0.5)  # Assuming loss ranges from 0 to 1
            medium = max(0, min((loss_value - 0.25) / 0.25, (0.75 - loss_value) / 0.25))
            high = max(0, (loss_value - 0.5) / 0.5)
            return low, medium, high
        
        def accuracy_membership(acc_value: float) -> Tuple[float, float, float]:  # Low, Medium, High
            low = max(0, (1 - acc_value) / 0.5)
            medium = max(0, min((acc_value - 0.25) / 0.25, (0.75 - acc_value) / 0.25))
            high = max(0, (acc_value - 0.5) / 0.5)
            return low, medium, high

        # Get fuzzy memberships
        loss_low, loss_medium, loss_high = loss_membership(loss)
        acc_low, acc_medium, acc_high = accuracy_membership(accuracy)

        # Apply fuzzy rules to determine the weight
        weight = (
            (loss_low * acc_high) +          # Low loss + High accuracy
            (loss_medium * acc_medium) +     # Medium loss + Medium accuracy
            (loss_high * acc_low)            # High loss + Low accuracy
        )
        
        return weight

    def aggregate_weights(self, weights_results):
        total_weight = sum(weight for _, weight in weights_results)
        normalized_weights_results = [
            (params, weight / total_weight) for params, weight in weights_results
        ]

        aggregated_ndarrays = aggregate(normalized_weights_results)
        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        metrics_aggregated = []  # You can fill this as needed

        return parameters_aggregated, metrics_aggregated
