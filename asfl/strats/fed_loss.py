
from flwr.common import ndarrays_to_parameters

### THIS IS FED AVG, BUT IT HAS DIFFERENT 

import numpy as np

from typing import Union, Callable, Dict, List, Optional, Tuple

from flwr.common import (
    FitRes,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy

from flwr.server.client_proxy import ClientProxy
from flwr.common.logger import log
from logging import WARNING, INFO, DEBUG, CRITICAL, ERROR
from .fed_custom import FedCustom
from .dat import aggregate

from functools import reduce


class FedLoss(FedCustom):
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

        metrics_list = [(client_proxy.cid, fit_res.metrics) for client_proxy, fit_res in results]

        # Assuming a small constant to avoid division by zero
        epsilon = 1e-6

        # Inverse of loss based weighting
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), 1 / (fit_res.metrics['loss'] + epsilon))
            for _, fit_res in results
        ]

        # Normalize the weights so they sum to 1
        total_weight = sum(weight for _, weight in weights_results)

        normalized_weights_results = [
            (params, weight / total_weight) for params, weight in weights_results
        ]

        aggregated_ndarrays = aggregate(normalized_weights_results)

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        metrics_aggregated = []
        
        return parameters_aggregated, metrics_aggregated