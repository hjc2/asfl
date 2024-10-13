
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

from flwr.common.logger import log
from logging import WARNING, INFO, DEBUG, CRITICAL, ERROR
from .fed_custom import FedCustom

from functools import reduce


class FedWide(FedCustom):

    def custom_eval(self, server_round, parameters, scalars):

        return (15,{"central_acc": 0.501})
    
    
    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        
        self.evaluate_fn = self.custom_eval


        log(CRITICAL, "Evaluation has ran!")
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            log(CRITICAL, "no evaluate fn found, nothing returned...")
            return None

        parameters_ndarrays = parameters_to_ndarrays(parameters)
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
        if eval_res is None:
            return None
        loss, metrics = eval_res
        log(CRITICAL, "eval found global" + str(loss))

        return loss, metrics