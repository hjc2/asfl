
from flwr.common import ndarrays_to_parameters
from typing import Union, Dict, List, Optional, Tuple
from flwr.common.logger import log
from logging import WARNING, INFO, DEBUG, CRITICAL, ERROR

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

class FedVariance(FedCustom):
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average of label variance."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        #Label Variance based weighting
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.metrics['labels'])
            for _, fit_res in results
        ]
            
        aggregated_ndarrays = aggregate(weights_results)

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        metrics_aggregated = []
        
        return parameters_aggregated, metrics_aggregated