
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

def adaptive_in_place(self, results: List[Tuple[ClientProxy, FitRes]], freq_appearance: Dict, server_round) -> NDArrays:
    """Compute in-place weighted average."""
    # Count total examples
    num_examples_total = sum(fit_res.num_examples for (_, fit_res) in results)
    advlog(self.adv_log, lambda: log(ERROR, f"num_examples_total : {num_examples_total}"))

    # Compute scaling factors for each result
    combined_factors = [
        ((fit_res.num_examples / num_examples_total) + ((((freq_appearance[client_proxy.cid] - 1) / server_round)) / len(results))) / 2 for client_proxy, fit_res in results
    ]

    freq_factors = [
        (((freq_appearance[client_proxy.cid] - 1) / server_round)) / len(results) for client_proxy, fit_res in results
    ]

    scaling_factors = [
        (fit_res.num_examples / num_examples_total) for _, fit_res in results
    ]

    eval_factors = average_lists(freq_factors, scaling_factors)

    for x, y, z, q in zip(scaling_factors, freq_factors, combined_factors, eval_factors):
        advlog(self.adv_log, lambda: log(WARNING, f"scaling: {x} freq: {y} combined: {z} eval: {q}"))

    # Let's do in-place aggregation
    # Get first result, then add up each other
    params = [
        eval_factors[0] * x for x in parameters_to_ndarrays(results[0][1].parameters)
    ]

    advlog(self.adv_log, lambda: log(WARNING, f"params: {params}"))

    for i, (_, fit_res) in enumerate(results[1:]):
        res = (
            scaling_factors[i + 1] * x
            for x in parameters_to_ndarrays(fit_res.parameters)
        )
        params = [reduce(np.add, layer_updates) for layer_updates in zip(params, res)]

    return params


class FedWide(FedCustom):
    # aggregates the training results
    # where the algo runs
    # "aggregate_fit is responsible for aggregating the results returned by the clients that were selected and asked to train in configure_fit."
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}
    
        freq_appearance = track_node_frequency(self.cid_ll) # RETURNS A COUNT OF HOW MANY ROUNDS IT WAS A PART OF


        aggregated_ndarrays = adaptive_in_place(self, results, freq_appearance, server_round)

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)
        
        metrics_aggregated = {}
        
        return parameters_aggregated, metrics_aggregated