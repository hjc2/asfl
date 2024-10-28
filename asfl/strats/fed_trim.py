
from flwr.common import ndarrays_to_parameters
import numpy as np
from typing import Union, Callable, Dict, List, Optional, Tuple
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.common.logger import log
from .fed_custom import FedCustom
from .dat import aggregate
from flwr.common.logger import log
from logging import WARNING, INFO, DEBUG, CRITICAL, ERROR

class FedTrim(FedCustom):
    def __init__(self, *, trim_fraction: float = 0.1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.trim_fraction = trim_fraction

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

        # Sort results based on loss
        results_sorted = sorted(results, key=lambda x: x[1].metrics['loss'])

        # Determine number of clients to trim
        num_clients_to_trim = int(len(results_sorted) * self.trim_fraction)
        trimmed_results = results_sorted[num_clients_to_trim:]

        if not trimmed_results:
            return None, {}

        # Aggregate metrics and prepare for weighted averaging
        metrics_list = [(client_proxy.cid, fit_res.metrics) for client_proxy, fit_res in trimmed_results]

        # Using a small epsilon to avoid division by zero
        epsilon = 1e-6

        # Inverse of loss-based weighting, with a sample size component
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), 
             (1 / (fit_res.metrics['loss'] + epsilon)) * fit_res.num_examples)
            for _, fit_res in trimmed_results
        ]

        # Normalize the weights
        total_weight = sum(weight for _, weight in weights_results)

        if total_weight == 0:
            return None, {}

        normalized_weights_results = [
            (params, weight / total_weight) for params, weight in weights_results
        ]

        # Aggregate parameters
        aggregated_ndarrays = aggregate(normalized_weights_results)
        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        # Aggregate metrics
        metrics_aggregated = {
            "trimmed_count": len(trimmed_results),
            "original_count": len(results),
            "trim_fraction": self.trim_fraction
        }

        log(INFO, f"Aggregated with trimmed mean. Trimmed count: {metrics_aggregated['trimmed_count']}, Original count: {metrics_aggregated['original_count']}")

        return parameters_aggregated, metrics_aggregated
