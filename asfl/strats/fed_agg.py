
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from asfl.task import Net, get_weights
from flwr.server.strategy import Strategy

### THIS IS FED AVG, BUT IT HAS DIFFERENT 

import numpy as np

from typing import Union, Callable, Dict, List, Optional, Tuple

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    NDArrays,
    MetricsAggregationFn,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion

from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.common.logger import log
from logging import WARNING, INFO, DEBUG, CRITICAL, ERROR
import random
from .logging_afl import CustomCriterion, FedCustom

from functools import reduce


WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_evaluate_clients`.
"""
def average_lists(*lists):
    if not lists:
        raise ValueError("At least one list must be provided.")
    
    list_lengths = [len(lst) for lst in lists]
    if len(set(list_lengths)) != 1:
        raise ValueError("All lists must have the same length.")
    
    num_lists = len(lists)
    list_length = len(lists[0])
    
    averages = [0.0] * list_length
    
    for i in range(list_length):
        total = sum(lst[i] for lst in lists)
        averages[i] = total / num_lists
    return averages

def track_node_appearances(data):
    last_appearance = {}
    for round_id, node_list in data[:-1]:
        for node_id in node_list:
            last_appearance[node_id] = round_id
    return last_appearance

def track_node_frequency(data):
    appearance_info = {}
    for _, node_list in data:
        for node_id in node_list:
            if node_id in appearance_info:
                appearance_info[node_id] = appearance_info[node_id] + 1
            else:
                appearance_info[node_id] = 1
    return appearance_info

def adaptive_in_place(results: List[Tuple[ClientProxy, FitRes]], freq_appearance: Dict, server_round) -> NDArrays:
    """Compute in-place weighted average."""
    # Count total examples
    num_examples_total = sum(fit_res.num_examples for (_, fit_res) in results)
    log(ERROR, f"num_examples_total : {num_examples_total}")

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
        log(WARNING, f"scaling: {x} freq: {y} combined: {z} eval: {q}")

    # Let's do in-place aggregation
    # Get first result, then add up each other
    params = [
        eval_factors[0] * x for x in parameters_to_ndarrays(results[0][1].parameters)
    ]

    log(WARNING, f"params: {params}")

    for i, (_, fit_res) in enumerate(results[1:]):
        res = (
            scaling_factors[i + 1] * x
            for x in parameters_to_ndarrays(fit_res.parameters)
        )
        params = [reduce(np.add, layer_updates) for layer_updates in zip(params, res)]

    return params

# FOR EACH OF THE MODELS
def adaptive_agg_fit(results: List[Tuple[NDArrays, int]], last_appearance, freq_appearance, server_round) -> NDArrays:
    
    """Compute weighted average."""
    # Calculate the total number of examples used during training
    num_examples_total = sum(num_examples for (_, num_examples, _ ) in results)

    for _, num_examples, cid in results:
        log(DEBUG, f"client: {cid} num_examples: {num_examples}")

    log(DEBUG, f"last appearance {last_appearance}")
    # for _, _, cid in results:
        # log(DEBUG, f"val: {last_appearance[cid]} round: {server_round}")

    weighted_weights = [
        [layer * num_examples for layer in weights] for weights, num_examples, cid in results
    ]

    # Compute average weights of each layer
    weights_prime: NDArrays = [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime

class FedAgg(FedCustom):
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
        
        # for client, _ in results:
            # log(DEBUG, f"Client: {client.node_id}")
        # log(DEBUG, f"cid_ll: {self.cid_ll}")
        freq_appearance = track_node_frequency(self.cid_ll) # RETURNS A COUNT OF HOW MANY ROUNDS IT WAS A PART OF

        if self.inplace:
            if(server_round == 1): # log only first time
                log(CRITICAL, "in place!")
            aggregated_ndarrays = adaptive_in_place(results, freq_appearance, server_round)
        else:
            # Convert results
            weights_results = [
                (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples, client.cid) for client, fit_res in results
            ]
            # my custom aggregation function
            last_appearance = track_node_appearances(self.cid_ll) # RETURNS A DICT OF [NODE ID] -> [LAST ROUND IT WAS SEEN (0 IF NEVER OR ROUND 1)]
            log(DEBUG, f"last_appearance: {last_appearance}")
            aggregated_ndarrays = adaptive_agg_fit(weights_results, last_appearance, freq_appearance, server_round)

        # resume other code

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)
        
        metrics_aggregated = {}
        
        return parameters_aggregated, metrics_aggregated