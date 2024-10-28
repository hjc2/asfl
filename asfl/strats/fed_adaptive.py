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
from .dat import aggregate, track_node_frequency
from .fed_custom import FedCustom

class FedAdaptive(FedCustom):
    def __init__(
        self,
        *args,
        num_samples_weight: float = 0.3,
        label_var_weight: float = 0.2,
        accuracy_weight: float = 0.3,
        freq_weight: float = 0.2,
        trim_fraction: float = 0.1,  # Fraction of models to trim
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.metric_weights = {
            "num_samples": num_samples_weight,
            "label_var": label_var_weight,
            "accuracy": accuracy_weight,
            "freq": freq_weight,
        }
        self.trim_fraction = trim_fraction  # Store the fraction for trimming
        self.client_history = {}
        self.round_metrics = {}

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model weights using adaptive weighting strategy with trimming of the bottom models."""
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        # Track client frequencies
        freq_appearance = track_node_frequency(self.cid_ll)

        # Calculate adaptive weights
        adaptive_weights = self._calculate_adaptive_weights(results, freq_appearance)

        # Extract parameters and num_examples
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), adaptive_weights[idx])
            for idx, (_, fit_res) in enumerate(results)
        ]

        # Aggregate parameters
        aggregated_weights = aggregate(weights_results)

        # Convert weights to parameters
        parameters_aggregated = ndarrays_to_parameters(aggregated_weights)

        # Calculate metrics for monitoring
        metrics = {
            "num_clients": len(results),
            "avg_weight": float(np.mean(adaptive_weights)),
            "std_weight": float(np.std(adaptive_weights)),
        }

        return parameters_aggregated, metrics
