from flwr.common import ndarrays_to_parameters
from typing import Union, Dict, List, Optional, Tuple
from flwr.common.logger import log
from logging import WARNING, INFO, DEBUG, CRITICAL, ERROR
import numpy as np

from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from .fed_custom import FedCustom
from .dat import aggregate

class FedProximal(FedCustom):
    def __init__(self, mu: float = 0.1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.mu = mu  # Proximal term coefficient

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using proximal averaging."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Collect parameters and local updates
        weights_results = []
        for client_proxy, fit_res in results:
            params_ndarrays = parameters_to_ndarrays(fit_res.parameters)
            # Weight by the number of examples
            weight = fit_res.num_examples
            weights_results.append((params_ndarrays, weight, fit_res.parameters))

        # Aggregate parameters with proximal term
        total_weight = sum(weight for _, weight, _ in weights_results)
        if total_weight == 0:
            return None, {}

        # Compute the aggregated parameters with proximal adjustment
        aggregated_ndarrays = []
        for param_ndarrays, weight, client_params in weights_results:
            if aggregated_ndarrays == []:
                aggregated_ndarrays = [np.zeros_like(param) for param in param_ndarrays]

            for i in range(len(param_ndarrays)):
                aggregated_ndarrays[i] += (weight / total_weight) * (param_ndarrays[i] - np.array(client_params[i])) + (self.mu * np.array(client_params[i]))

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        # Optional: Aggregate metrics if needed
        metrics_aggregated = {
            "total_clients": len(results),
            "total_weight": total_weight
        }

        log(INFO, f"Aggregated using FedProx. Total clients: {metrics_aggregated['total_clients']}, Total weight: {metrics_aggregated['total_weight']}")

        return parameters_aggregated, metrics_aggregated
