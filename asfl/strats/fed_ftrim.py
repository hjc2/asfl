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
from .fed_adaptive import FedAdaptive

class FuzzySet:
    def __init__(self, low: float, mid: float, high: float):
        self.low = low
        self.mid = mid
        self.high = high
    
    def membership(self, x: float) -> float:
        """Calculate triangular membership function."""
        if x <= self.low or x >= self.high:
            return 0.0
        elif x <= self.mid:
            return (x - self.low) / (self.mid - self.low)
        else:
            return (self.high - x) / (self.high - self.mid)

class FedFtrim(FedAdaptive):
    def __init__(
        self,
        *args,
        trim_fraction: float = 0.1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.trim_fraction = trim_fraction
        self.client_history = {}
        self.round_metrics = {}
        
        # Define fuzzy sets for each metric
        self.fuzzy_sets = {
            "num_samples": {
                "low": FuzzySet(0, 100, 500),
                "medium": FuzzySet(300, 750, 1200),
                "high": FuzzySet(1000, 2000, 3000)
            },
            "label_var": {
                "low": FuzzySet(0, 0.1, 0.3),
                "medium": FuzzySet(0.2, 0.4, 0.6),
                "high": FuzzySet(0.5, 0.7, 1.0)
            },
            "accuracy": {
                "low": FuzzySet(0, 0.3, 0.5),
                "medium": FuzzySet(0.4, 0.6, 0.8),
                "high": FuzzySet(0.7, 0.85, 1.0)
            },
            "freq": {
                "low": FuzzySet(0, 0.2, 0.4),
                "medium": FuzzySet(0.3, 0.5, 0.7),
                "high": FuzzySet(0.6, 0.8, 1.0)
            }
        }
        
        # Define fuzzy rules weights (importance of each category)
        self.rule_weights = {
            "num_samples": 0.3,
            "label_var": 0.2,
            "accuracy": 0.3,
            "freq": 0.2
        }

    def _calculate_fuzzy_weight(self, metric_name: str, value: float) -> float:
        """Calculate fuzzy weight for a single metric."""
        sets = self.fuzzy_sets[metric_name]
        memberships = {
            level: fuzzy_set.membership(value)
            for level, fuzzy_set in sets.items()
        }
        
        # Weighted sum of memberships
        weight_factors = {
            "low": 0.2,
            "medium": 0.5,
            "high": 1.0
        }
        
        total_membership = sum(memberships.values())
        if total_membership == 0:
            return 0.5  # Default middle value if no membership
        
        weighted_sum = sum(
            memberships[level] * weight_factors[level]
            for level in memberships
        )
        
        return weighted_sum / total_membership

    def _calculate_adaptive_weights(
        self,
        results: List[Tuple[ClientProxy, FitRes]],
        freq_appearance: Dict[str, float]
    ) -> np.ndarray:
        """Calculate adaptive weights using fuzzy logic."""
        weights = []
        
        for client_proxy, fit_res in results:
            metrics = {
                "num_samples": float(fit_res.num_examples),
                "label_var": float(fit_res.metrics.get("label_variance", 0.5)),
                "accuracy": float(fit_res.metrics.get("accuracy", 0.0)),
                "freq": float(freq_appearance.get(client_proxy.cid, 0.0))
            }
            
            # Calculate fuzzy weights for each metric
            fuzzy_weights = {
                metric: self._calculate_fuzzy_weight(metric, value)
                for metric, value in metrics.items()
            }
            
            # Combine fuzzy weights using rule weights
            final_weight = sum(
                fuzzy_weights[metric] * self.rule_weights[metric]
                for metric in fuzzy_weights
            )
            
            weights.append(final_weight)
        
        # Normalize weights
        weights = np.array(weights)
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            weights = np.ones_like(weights) / len(weights)
            
        return weights

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model weights using fuzzy logic-based adaptive weighting strategy."""
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        # Track client frequencies
        freq_appearance = track_node_frequency(self.cid_ll)

        # Sort results by accuracy
        sorted_results = sorted(
            results,
            key=lambda x: x[1].metrics.get("accuracy", 0.0),
            reverse=True
        )

        # Calculate the number of models to trim
        num_to_trim = max(1, int(len(sorted_results) * self.trim_fraction))

        # Filter out the bottom fraction of models
        trimmed_results = sorted_results[:-num_to_trim]

        if not trimmed_results:
            return None, {}

        # Calculate adaptive weights using fuzzy logic
        adaptive_weights = self._calculate_adaptive_weights(trimmed_results, freq_appearance)

        # Extract parameters and num_examples
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), adaptive_weights[idx])
            for idx, (_, fit_res) in enumerate(trimmed_results)
        ]

        # Aggregate parameters
        aggregated_weights = aggregate(weights_results)

        # Convert weights to parameters
        parameters_aggregated = ndarrays_to_parameters(aggregated_weights)

        # Calculate metrics for monitoring
        metrics = {
            "num_clients": len(trimmed_results),
            "avg_weight": float(np.mean(adaptive_weights)),
            "std_weight": float(np.std(adaptive_weights)),
            "max_weight": float(np.max(adaptive_weights)),
            "min_weight": float(np.min(adaptive_weights))
        }

        return parameters_aggregated, metrics