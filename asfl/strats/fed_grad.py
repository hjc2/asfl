
from flwr.common import ndarrays_to_parameters

### THIS IS FED AVG, BUT IT HAS DIFFERENT 

from typing import Union, Dict, List, Optional, Tuple

from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.common.logger import log
from logging import WARNING, INFO, DEBUG, CRITICAL, ERROR
from .fed_custom import FedCustom
from .dat import aggregate
import numpy as np
from typing import List, Dict, Union
from scipy.spatial.distance import cosine

class FedGrad(FedCustom):


    def compare_parameters(params1: List[np.ndarray], params2: List[np.ndarray]) -> Dict[str, Union[float, List[float]]]:
        """
        Compare two sets of parameters (NDArrays) and return various similarity metrics.
        
        Args:
        params1 (List[np.ndarray]): First set of parameters
        params2 (List[np.ndarray]): Second set of parameters
        
        Returns:
        Dict[str, Union[float, List[float]]]: Dictionary containing similarity metrics
        """
        if len(params1) != len(params2):
            raise ValueError("The two parameter sets must have the same number of arrays")
        
        results = {}
        
        # Flatten and concatenate all arrays
        flat_params1 = np.concatenate([arr.flatten() for arr in params1])
        flat_params2 = np.concatenate([arr.flatten() for arr in params2])
        
        # Cosine similarity
        cos_sim = 1 - cosine(flat_params1, flat_params2)
        results['cosine_similarity'] = cos_sim
        
        # Euclidean distance
        eucl_dist = np.linalg.norm(flat_params1 - flat_params2)
        results['euclidean_distance'] = eucl_dist
        
        # Normalized Euclidean distance (to account for different scales)
        norm_eucl_dist = eucl_dist / (np.linalg.norm(flat_params1) + np.linalg.norm(flat_params2))
        results['normalized_euclidean_distance'] = norm_eucl_dist
        
        # Element-wise absolute difference
        abs_diff = np.abs(flat_params1 - flat_params2)
        results['mean_absolute_difference'] = np.mean(abs_diff)
        results['max_absolute_difference'] = np.max(abs_diff)
        
        # Layer-wise comparisons
        layer_cos_sim = []
        layer_eucl_dist = []
        layer_mean_abs_diff = []
        for arr1, arr2 in zip(params1, params2):
            flat1 = arr1.flatten()
            flat2 = arr2.flatten()
            layer_cos_sim.append(1 - cosine(flat1, flat2))
            layer_eucl_dist.append(np.linalg.norm(flat1 - flat2))
            layer_mean_abs_diff.append(np.mean(np.abs(flat1 - flat2)))
        
        results['layer_cosine_similarity'] = layer_cos_sim
        results['layer_euclidean_distance'] = layer_eucl_dist
        results['layer_mean_absolute_difference'] = layer_mean_abs_diff
        
        return results

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

        metrics_list = [fit_res.parameters for _, fit_res in results]

        if self.parameter_history is not None:
            history_ndarrays = parameters_to_ndarrays(self.parameter_history)

            compared_results = [
                self.compare_parameters(parameters_to_ndarrays(fit_res.parameters), history_ndarrays)
                  for _, fit_res in results
                ]

            weights_results = [
                (parameters_to_ndarrays(fit_res.parameters), compared_res['layer_cosine_similarity'])
                for compared_res, (_, fit_res) in zip(compared_results, results)
            ]

        else:
            #Num Examples for First
            weights_results = [
                (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                for _, fit_res in results
            ]

        aggregated_ndarrays = aggregate(weights_results)

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        self.parameter_history = aggregated_ndarrays # set the parameter history to the aggregated parameters of this round

        metrics_aggregated = []
        
        return parameters_aggregated, metrics_aggregated