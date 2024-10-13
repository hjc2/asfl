
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
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.common.logger import log
from logging import WARNING, INFO, DEBUG, CRITICAL, ERROR
from .fed_custom import FedCustom
from .dat import average_lists, track_node_frequency, track_node_appearances, advlog

from functools import reduce


class FedAcc(FedCustom):
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # structure [(CID, {accuracy, loss})]
        metrics_list = [(client_proxy.cid, fit_res.metrics) for client_proxy, fit_res in results]

        #Accuracy based weighting
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.metrics['accuracy'])
            for _, fit_res in results
        ]

        aggregated_ndarrays = aggregate(weights_results)

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        metrics_aggregated = []
        
        return parameters_aggregated, metrics_aggregated