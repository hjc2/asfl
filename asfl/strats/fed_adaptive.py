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
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.metric_weights = {
            "num_samples": num_samples_weight,
            "label_var": label_var_weight,
            "accuracy": accuracy_weight,
            "freq": freq_weight,
        }
        self.client_history = {}
        self.round_metrics = {}

    def _normalize_metric(self, values: List[float]) -> List[float]:
        """Normalize values to range [0, 1]."""
        min_val = min(values)
        max_val = max(values)
        if max_val == min_val:
            return [1.0] * len(values)
        return [(v - min_val) / (max_val - min_val) for v in values]

    def _calculate_adaptive_weights(
        self,
        results: List[Tuple[ClientProxy, FitRes]],
        freq_appearance: Dict[str, int],
    ) -> List[float]:
        """Calculate adaptive weights based on multiple metrics."""
        metrics = {
            "num_samples": [],
            "label_var": [],
            "accuracy": [],
            "freq": [],
        }

        # Extract metrics from results
        for client_proxy, fit_res in results:
            client_id = client_proxy.cid
            metrics["num_samples"].append(fit_res.num_examples)
            metrics["label_var"].append(fit_res.metrics.get("label_variance", 0.0))
            metrics["accuracy"].append(fit_res.metrics.get("accuracy", 0.0))
            metrics["freq"].append(freq_appearance.get(client_id, 0))

        # Normalize all metrics
        normalized_metrics = {
            k: self._normalize_metric(v) for k, v in metrics.items()
        }

        # Calculate final weights
        weights = np.zeros(len(results))
        for metric_name, norm_values in normalized_metrics.items():
            if metric_name == "label_var":
                # Inverse weight for label variance (lower is better)
                norm_values = [1 - v for v in norm_values]
            weights += np.array(norm_values) * self.metric_weights[metric_name]

        # Normalize final weights
        weights = weights / np.sum(weights)
        return weights.tolist()

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model weights using adaptive weighting strategy."""
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