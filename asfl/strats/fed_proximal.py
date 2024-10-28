
from flwr.common import ndarrays_to_parameters

### THIS IS FED AVG, BUT IT HAS DIFFERENT 

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

class FedProximal(FedCustom):
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

        #Accuracy based weighting
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
            
        aggregated_ndarrays = aggregate(weights_results)

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        metrics_aggregated = []
        
        return parameters_aggregated, metrics_aggregated
    