# ############################
# ## NOTES ON IMPORT FORMAT ##
# ############################
#
# From https://github.com/dagster-io/dagster/blob/master/python_modules/dagster/dagster/__init__.py
#
# This file defines your package's public API. Imports need to be structured/formatted so as to to ensure
# that the broadest possible set of static analyzers understand your_package's public API as intended.
# The below guidelines ensure this is the case.
#
# (1) All imports in this module intended to define exported symbols should be of the form `from
# your_package.foo import X as X`. This is because imported symbols are not by default considered public
# by static analyzers. The redundant alias form `import X as X` overwrites the private imported `X`
# with a public `X` bound to the same value. It is also possible to expose `X` as public by listing
# it inside `__all__`, but the redundant alias form is preferred here due to easier maintainability.

# (2) All imports should target the module in which a symbol is actually defined, rather than a
# container module where it is imported. This rule also derives from the default private status of
# imported symbols. So long as there is a private import somewhere in the import chain leading from
# an import to its definition, some linters will be triggered (e.g. pyright). For example, the
# following results in a linter error when using your_package as a third-party library:

#     ### your_package/foo/bar.py
#     BAR = "BAR"
#
#     ### your_package/foo/__init__.py
#     from .bar import BAR  # BAR is imported so it is not part of your_package.foo public interface
#     FOO = "FOO"
#
#     ### your_package/__init__.py
#     from .foo import FOO, BAR  # importing BAR is importing a private symbol from your_package.foo
#     __all__ = ["FOO", "BAR"]
#
#     ### some_user_code.py
#     # from your_package import BAR  # linter error even though `BAR` is in `your_package.__all__`!
#
# We could get around this by always remembering to use the `from .foo import X as X` form in
# containers, but it is simpler to just import directly from the defining module.

# Specs
from .specs.value import ValueFrame as ValueFrame
from .specs.from_legacy import PredictorGroupSpec as PredictorGroupSpec
from .specs.outcome import OutcomeSpec as OutcomeSpec, BooleanOutcomeSpec as BooleanOutcomeSpec
from .specs.prediction_times import PredictionTimeFrame as PredictionTimeFrame
from .specs.temporal import PredictorSpec as PredictorSpec
from .specs.static import StaticSpec as StaticSpec, StaticFrame as StaticFrame
from .specs.timedelta import TimeDeltaSpec as TimeDeltaSpec
from .specs.timestamp import TimestampValueFrame as TimestampValueFrame

# Aggregators
from .aggregators import (
    MaxAggregator as MaxAggregator,
    MeanAggregator as MeanAggregator,
    MinAggregator as MinAggregator,
    SlopeAggregator as SlopeAggregator,
    SumAggregator as SumAggregator,
    VarianceAggregator as VarianceAggregator,
    HasValuesAggregator as HasValuesAggregator,
    CountAggregator as CountAggregator,
)

# Main
from .main import Flattener as Flattener
from .main import ValueSpecification as ValueSpecification
