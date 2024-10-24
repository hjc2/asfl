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
from .dat import aggregate
from .fed_fuzz import FedFuzz

class FedDouble(FedFuzz):
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
            num_samples = fit_res.metrics['num_samples']
            label_variance = fit_res.metrics['labels']
            weight = self.calculate_fuzzy_weight(num_samples, label_variance)

            weights_results.append((parameters_to_ndarrays(fit_res.parameters), weight))

        return self.aggregate_weights(weights_results)
