

import numpy as np
from typing import List, Dict, Tuple, Optional
from flwr.common import Parameters, Scalar, NDArrays
from .fed_custom import FedCustom
from flwr.server.client_proxy import ClientProxy
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
    parameters_to_ndarrays,
    ndarrays_to_parameters,

)

class FedTrust(FedCustom):
    def __init__(
        self,
        *args,
        trust_alpha: float = 0.8,        # Trust score decay
        momentum: float = 0.9,           # Momentum factor
        adapt_rate: float = 0.1,         # Adaptation rate
        dampening: float = 0.5,          # Initial dampening factor
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.trust_scores = {}           # Client trust scores
        self.momentum_buffer = None      # Momentum buffer
        self.previous_updates = {}       # Track previous updates
        self.dampening = dampening      
        self.trust_alpha = trust_alpha
        self.momentum = momentum
        self.adapt_rate = adapt_rate
        self.current_weights = None
        self.round_number = 0

    def calculate_trust_score(
        self,
        client_id: str,
        current_update: NDArrays,
        previous_update: Optional[NDArrays]
    ) -> float:
        """Calculate trust score based on update consistency and magnitude."""
        if previous_update is None:
            return 1.0  # Initial trust
            
        # Calculate update similarity using cosine similarity
        def flatten_and_concatenate(arrays):
            return np.concatenate([arr.flatten() for arr in arrays])
            
        current_flat = flatten_and_concatenate(current_update)
        previous_flat = flatten_and_concatenate(previous_update)
        
        # Cosine similarity
        similarity = np.dot(current_flat, previous_flat) / (
            np.linalg.norm(current_flat) * np.linalg.norm(previous_flat)
        )
        
        # Update magnitude ratio
        curr_magnitude = np.linalg.norm(current_flat)
        prev_magnitude = np.linalg.norm(previous_flat)
        magnitude_ratio = min(curr_magnitude, prev_magnitude) / max(curr_magnitude, prev_magnitude)
        
        # Combined score
        update_score = 0.7 * similarity + 0.3 * magnitude_ratio
        
        # Exponential moving average of trust
        current_trust = self.trust_scores.get(client_id, 1.0)
        new_trust = self.trust_alpha * current_trust + (1 - self.trust_alpha) * update_score
        
        return new_trust

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate using trust scores and adaptive momentum."""
        self.round_number += 1
        
        if not results:
            return None, {}
            
        # Extract weights and calculate updates
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples, client.cid)
            for client, fit_res in results
        ]
        
        # Calculate trust-weighted updates
        trusted_updates = []
        total_trust = 0
        
        for weights, num_examples, client_id in weights_results:
            # Calculate client's update
            if self.current_weights is None:
                self.current_weights = weights
                continue
                
            client_update = [w - cw for w, cw in zip(weights, self.current_weights)]
            
            # Calculate trust score
            trust_score = self.calculate_trust_score(
                client_id,
                client_update,
                self.previous_updates.get(client_id)
            )
            
            # Store current update for next round
            self.previous_updates[client_id] = client_update
            self.trust_scores[client_id] = trust_score
            
            # Weight update by trust and number of examples
            weighted_update = [
                trust_score * num_examples * update
                for update in client_update
            ]
            trusted_updates.append(weighted_update)
            total_trust += trust_score * num_examples
            
        # Normalize updates
        averaged_update = [
            sum(update[i] for update in trusted_updates) / total_trust
            for i in range(len(trusted_updates[0]))
        ]
        
        # Update dampening factor based on update magnitude
        update_magnitude = np.mean([
            np.linalg.norm(layer) 
            for layer in averaged_update
        ])
        self.dampening *= (1 - self.adapt_rate * (1 - np.tanh(update_magnitude)))
        
        # Apply momentum with adaptive dampening
        if self.momentum_buffer is None:
            self.momentum_buffer = averaged_update
        else:
            self.momentum_buffer = [
                self.momentum * mb + (1 - self.momentum) * update
                for mb, update in zip(self.momentum_buffer, averaged_update)
            ]
        
        # Apply dampened momentum update
        new_weights = [
            w + self.dampening * mb
            for w, mb in zip(self.current_weights, self.momentum_buffer)
        ]
        
        self.current_weights = new_weights
        
        return ndarrays_to_parameters(self.current_weights), {}