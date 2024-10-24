
from flwr.common import (
    FitRes,
    NDArrays,
)
from typing import Dict, List, Optional, Tuple
from functools import reduce
import numpy as np
from flwr.common.logger import log
from logging import WARNING, INFO, DEBUG, CRITICAL, ERROR
def aggregate(results: List[Tuple[NDArrays, int]]) -> NDArrays:
    """Compute weighted average."""
    # Calculate the total number of examples used during training
    total_weights = sum(weight_value for (_, weight_value) in results)

    weighted_test = [
        weight for _, weight in results
    ]
    # log(INFO, "weighted test: " + str(weighted_test))

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * num_examples for layer in weights] for weights, num_examples in results
    ]

    # Compute average weights of each layer
    weights_prime: NDArrays = [
        reduce(np.add, layer_updates) / total_weights
        for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime

# provides the averages of several lists
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

# tracks the number of appearances in a set of nodes
def track_node_appearances(data):
    last_appearance = {}
    for round_id, node_list in data[:-1]:
        for node_id in node_list:
            last_appearance[node_id] = round_id
    return last_appearance

# tracks the number of occurences for a set of nodes
def track_node_frequency(data):
    appearance_info = {}
    for _, node_list in data:
        for node_id in node_list:
            if node_id in appearance_info:
                appearance_info[node_id] = appearance_info[node_id] + 1
            else:
                appearance_info[node_id] = 1
    return appearance_info

def advlog(adv_log, func):
    if(adv_log):
        return func()