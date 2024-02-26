"""Init timeseriesflattener."""
from .feature_specs.group_specs import PredictorGroupSpec
from .feature_specs.single_specs import OutcomeSpec, PredictorSpec
from .feature_specs.group_specs import OutcomeGroupSpec
from .flattened_dataset import TimeseriesFlattener
