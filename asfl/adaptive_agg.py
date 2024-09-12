
from typing import Dict, List, Optional, Tuple, Union
from flwr.server.client_proxy import ClientProxy
from flwr.common import (
    EvaluateRes,
    Scalar,
    NDArrays,
)
from logging import WARNING
from flwr.common.logger import log
from functools import reduce
import numpy as np

from flwr.server.strategy.aggregate import aggregate, aggregate_inplace, weighted_loss_avg

# FOR EACH OF THE MODELS
def adaptive_agg_fit(results: List[Tuple[NDArrays, int]]) -> NDArrays:
    
    """Compute weighted average."""
    # Calculate the total number of examples used during training
    num_examples_total = sum(num_examples for (_, num_examples) in results)

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * num_examples for layer in weights] for weights, num_examples in results
    ]

    # Compute average weights of each layer
    weights_prime: NDArrays = [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime