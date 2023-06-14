"""Templates for feature specifications."""
import logging
import time
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
from attr import dataclass
from pydantic import Field
from timeseriesflattener.feature_specs.single_specs import AnySpec
from timeseriesflattener.misc_utils import data_loaders, split_dfs
from timeseriesflattener.aggregation_functions import resolve_multiple_fns

log = logging.getLogger(__name__)
