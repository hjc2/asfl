
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


