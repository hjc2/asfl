
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

def adaptive_agg(results: List[Tuple[int, float]]) -> float:
    """Aggregate evaluation results obtained from multiple clients."""
    num_total_evaluation_examples = sum(num_examples for (num_examples, _) in results)
    weighted_losses = [num_examples * loss for num_examples, loss in results]
    return sum(weighted_losses) / num_total_evaluation_examples
    # return 1.0


def adapt_aggregate_evaluate(
    self,
    server_round: int,
    results: List[Tuple[ClientProxy, EvaluateRes]],
    failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
) -> Tuple[Optional[float], Dict[str, Scalar]]:
    """Aggregate evaluation losses using weighted average."""
    if not results:
        return None, {}
    # Do not aggregate if there are failures and failures are not accepted
    if not self.accept_failures and failures:
        return None, {}

    # Aggregate loss
    loss_aggregated = adaptive_agg(
        [
            (evaluate_res.num_examples, evaluate_res.loss)
            for _, evaluate_res in results
        ]
    )

    metrics_aggregated = {}

        
    # ACCURACY CALCULATIONS
    accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
    examples = [r.num_examples for _, r in results]
    aggregated_accuracy = sum(accuracies) / sum(examples)

    metrics_aggregated["accuracy"] = aggregated_accuracy
    metrics_aggregated["count"] = len(results)

    return loss_aggregated, metrics_aggregated


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
