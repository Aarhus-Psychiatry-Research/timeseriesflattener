"""Templates for feature specifications."""
import itertools
import logging
import time
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import pandas as pd
from frozendict import frozendict
from pydantic import BaseModel as PydanticBaseModel
from pydantic import Extra, Field

from timeseriesflattener.resolve_multiple_functions import resolve_multiple_fns
from timeseriesflattener.utils import data_loaders, split_dfs

log = logging.getLogger(__name__)

# pylint: disable=consider-alternative-union-syntax, trailing-whitespace, missing-class-docstring, too-few-public-methods


@lru_cache
def load_df_with_cache(
    loader_fn: Callable,
    kwargs: Dict[str, Any],
    feature_name: str,
) -> pd.DataFrame:
    """Wrapper function to cache dataframe loading."""
    start_time = time.time()
    log.info(
        f"{feature_name}: Loading values",
    )  # pylint: disable=logging-fstring-interpolation

    df = loader_fn(**kwargs)

    end_time = time.time()

    log.debug(
        f"{feature_name}: Loaded in {end_time - start_time:.2f} seconds",
    )  # pylint: disable=logging-fstring-interpolation

    return df


def in_dict_and_not_none(d: dict, key: str) -> bool:
    """Check if a key is in a dictionary and its value is not None."""
    return key in d and d[key] is not None


def resolve_from_dict_or_registry(data: Dict[str, Any]):
    """Resolve values_df from a dictionary or registry."""

    if "values_name" in data and data["values_name"] is not None:
        log.info(f"Resolving values_df from {data['values_name']}")
        data["values_df"] = split_dfs.get(data["values_name"])
        data["feature_name"] = data["values_name"]
    else:
        if isinstance(data["values_loader"], str):
            data["feature_name"] = data["values_loader"]
            data["values_loader"] = data_loaders.get(data["values_loader"])

        if callable(data["values_loader"]):
            if "loader_kwargs" not in data or data["loader_kwargs"] is None:
                data["loader_kwargs"] = {}

            data["values_df"] = load_df_with_cache(
                loader_fn=data["values_loader"],
                kwargs=frozendict(data["loader_kwargs"]),  # type: ignore
                feature_name=data["feature_name"],
            )


def resolve_values_df(data: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """Resolve the values_df attribute to a dataframe."""
    if not any(key in data for key in ["values_loader", "values_name", "values_df"]):
        raise ValueError(
            "Either values_loader or a dictionary containing dataframes or a single dataframe must be specified.",
        )

    if (
        sum(
            in_dict_and_not_none(data, key)
            for key in ["values_loader", "values_name", "values_df"]
        )
        > 1
    ):
        raise ValueError(
            "Only one of values_loader or values_name or df can be specified.",
        )

    if "values_df" not in data or data["values_df"] is None:
        resolve_from_dict_or_registry(data)

    if not isinstance(data["values_df"], pd.DataFrame):
        raise ValueError("values_df must be or resolve to a pandas DataFrame.")

    return data


class BaseModel(PydanticBaseModel):
    """Modified Pydantic BaseModel to allow arbitrary

    types and disallow attributes not in the class.
    """

    # The docstring generator uses the `short_description` attribute of the `Doc`
    # class to generate the top of the docstring.
    # If you want to modify a docstring, modify the `short_description` attribute.
    # Then, when you run tests, new docstrings will be generated which you can copy/paste into
    # the relevant files. This is necessary because
    # 1) we have inheritance and don't want to have one source of truth for docs
    # 2) pylance reads the docstring directly from the static file
    # This means we want to auto-generate docstrings to support the inheritance,
    # but also need to hard-code the docstring to support pylance.
    class Doc:
        short_description: str = """Modified Pydantic BaseModel to allow arbitrary
        types and disallow attributes not in the class."""

    class Config:
        """Disallow  attributes not in the the class."""

        arbitrary_types_allowed = True
        extra = Extra.forbid


def generate_docstring_from_attributes(cls: BaseModel) -> str:
    """Generate a docstring from the attributes of a Pydantic basemodel.

    The top of the docstring is taken from the `short_description` attribute of the `Doc`
    class. The rest of the docstring is generated from the attributes of the class.
    """
    doc = ""
    doc += f"{cls.Doc.short_description}\n\n    "
    doc += "Fields:\n"
    for field_name, field_obj in cls.__fields__.items():
        # extract the pretty printed type
        # __repr_args__ returns a list of tuples with two values,
        # the name of the argument and the value. We are only interested in the
        # value of the type argument.
        type_ = [arg[1] for arg in field_obj.__repr_args__() if arg[0] == "type"]
        type_ = type_[0]

        field_description = field_obj.field_info.description

        default_value = field_obj.default
        default_str = (
            f"Defaults to: {default_value}." if default_value is not None else ""
        )
        # Whitespace added for formatting
        doc += "        "
        doc += f"{field_name} ({type_}):\n        "

        doc += f"    {field_description} {default_str}\n"
    # remove the last newline
    doc = doc[:-1]
    return doc


def check_that_col_names_in_kwargs_exist_in_df(data: Dict[str, Any], df: pd.DataFrame):
    """Check that all column names in data are in the dataframe.

    Keys with "col_name" in them specify a column name.

    The dataframe should be in the values_df key of data.
    """
    attributes_with_col_name = {
        key for key in data if "col_name" in key and isinstance(data[key], str)
    }

    skip_attributes = {"output_col_name_override"}

    attributes_to_test = attributes_with_col_name - skip_attributes

    errors = []

    for attribute_key in attributes_to_test:
        col_name = data[attribute_key]

        if col_name not in df.columns:
            errors.append(f"{attribute_key}: {col_name} is not in df")

    if len(errors) > 0:
        raise ValueError("\n".join(errors))


class _AnySpec(BaseModel):
    """A base class for all feature specifications.

    Fields:
        values_loader (Optional[Callable]):
            Loader for the df. Tries to resolve from the data_loaders registry,
            then calls the function which should return a dataframe.
        values_name (Optional[str]):
            A string that maps to a key in a dictionary instantiated by
            `split_df_and_register_to_dict`. Each key corresponds to a dataframe, which
            is a subset of the df where the values_name == key.
        loader_kwargs (Optional[Mapping[str, Any]]):
            Optional kwargs for the values_loader.
        values_df (Optional[DataFrame]):
            Dataframe with the values.
        feature_name (str):
            The name of the feature. Used for column name generation, e.g.
            <prefix>_<feature_name>.
        prefix (str):
            The prefix used for column name generation, e.g.
            <prefix>_<feature_name>.
        input_col_name_override (Optional[str]):
            An override for the input column name. If None, will  attempt
            to infer it by looking for the only column that doesn't match id_col_name
            or timestamp_col_name.
        output_col_name_override (Optional[str]):
    Override the generated column name after flattening the time series
    """

    class Doc:
        short_description = "A base class for all feature specifications."

    values_loader: Optional[Callable] = Field(
        None,
        description="""Loader for the df. Tries to resolve from the data_loaders registry,
            then calls the function which should return a dataframe.""",
    )

    values_name: Optional[str] = Field(
        default=None,
        description="""A string that maps to a key in a dictionary instantiated by
            `split_df_and_register_to_dict`. Each key corresponds to a dataframe, which
            is a subset of the df where the values_name == key.""",
    )

    loader_kwargs: Optional[Dict[str, Any]] = Field(
        default=None,
        description="""Optional kwargs for the values_loader.""",
    )

    values_df: Optional[pd.DataFrame] = Field(
        default=None,
        description="Dataframe with the values.",
    )

    feature_name: str = Field(
        description="""The name of the feature. Used for column name generation, e.g.
            <prefix>_<feature_name>.""",
    )

    prefix: str = Field(
        description="""The prefix used for column name generation, e.g.
            <prefix>_<feature_name>.""",
    )

    input_col_name_override: Optional[str] = Field(
        default=None,
        description="""An override for the input column name. If None, will  attempt
            to infer it by looking for the only column that doesn't match id_col_name
            or timestamp_col_name.""",
    )

    output_col_name_override: Optional[str] = Field(
        default=None,
        description="""Override the generated column name after flattening the time series""",
    )

    def __init__(self, **kwargs: Any):
        kwargs = resolve_values_df(kwargs)

        # Check that required columns exist
        check_that_col_names_in_kwargs_exist_in_df(kwargs, df=kwargs["values_df"])

        if (
            "input_col_name_override" not in kwargs
            and "value" not in kwargs["values_df"].columns
        ):
            raise KeyError(
                f"The values_df must have a column named 'value' or an input_col_name_override must be specified. Columns in values_df: {list(kwargs['values_df'].columns)}",
            )

        if in_dict_and_not_none(d=kwargs, key="output_col_name_override"):
            # If an output_col_name_override is specified, don't prepend a prefix to it
            kwargs["prefix"] = ""

        super().__init__(**kwargs)

        # Type-hint the values_df to no longer be optional. Changes the outwards-facing
        # type hint so that mypy doesn't complain.
        self.values_df: pd.DataFrame = self.values_df

    def get_col_str(self, additional_feature_name: Optional[str] = None) -> str:
        """Create column name for the output column.

        Args:
            additional_feature_name (Optional[str]): An additional feature name
                to append to the column name.
        """
        feature_name = self.feature_name
        if additional_feature_name:
            feature_name = f"{feature_name}-{additional_feature_name}"
        col_str = f"{self.prefix}_{feature_name}"
        return col_str

    def __eq__(self, other: Any) -> bool:
        """Add equality check for dataframes.

        Trying to run `spec in list_of_specs` works for all attributes except for df, since the truth value of a dataframe is ambiguous.
        To remedy this, we use pandas' .equals() method for comparing the dfs, and get the combined truth value.
        """
        try:
            other_attributes_equal = all(
                getattr(self, attr) == getattr(other, attr)
                for attr in self.__dict__
                if attr != "values_df"
            )
        except AttributeError:
            return False

        dfs_equal = self.values_df.equals(other.values_df)  # type: ignore

        return other_attributes_equal and dfs_equal


class StaticSpec(_AnySpec):
    class Doc:
        short_description = """Specification for a static feature."""


class TemporalSpec(_AnySpec):
    """The minimum specification required for collapsing a temporal
        feature, whether looking ahead or behind. Mostly used for inheritance below.

    Fields:
        values_loader (Optional[Callable]):
            Loader for the df. Tries to resolve from the data_loaders registry,
            then calls the function which should return a dataframe.
        values_name (Optional[str]):
            A string that maps to a key in a dictionary instantiated by
            `split_df_and_register_to_dict`. Each key corresponds to a dataframe, which
            is a subset of the df where the values_name == key.
        loader_kwargs (Optional[dict]):
            Optional kwargs passed onto the data loader.
        values_df (Optional[DataFrame]):
            Dataframe with the values.
        feature_name (str):
            The name of the feature. Used for column name generation, e.g.
            <prefix>_<feature_name>.
        prefix (str):
            The prefix used for column name generation, e.g.
            <prefix>_<feature_name>.
        input_col_name_override (Optional[str]):
            An override for the input column name. If None, will  attempt
            to infer it by looking for the only column that doesn't match id_col_name
            or timestamp_col_name.
        output_col_name_override (Optional[str]):
            Override the generated column name after flattening the time series
        interval_days (Union[int, float]):
            How far to look in the given direction (ahead for outcomes,
            behind for predictors)
        resolve_multiple_fn (Union[Callable, str]):
            A function used for resolving multiple values within the
            interval_days.
        key_for_resolve_multiple (Optional[str]):
            Key used to lookup the resolve_multiple_fn in the
            resolve_multiple_fns registry. Used for column name generation. Only
            required if you don't specify a resolve_multiple_fn. Call
            timeseriesflattener.resolve_multiple_fns.resolve_multiple_fns.get_all()
            for a list of options.
        fallback (Union[Callable, int, float, str]):
            Which value to use if no values are found within interval_days.
        allowed_nan_value_prop (float):
            If NaN is higher than this in the input dataframe during
            resolution, raise an error. Defaults to: 0.0.
        entity_id_col_name (str):
            Col name for ids in the input dataframe. Defaults to: entity_id.
    """

    class Doc:
        short_description = """The minimum specification required for collapsing a temporal
        feature, whether looking ahead or behind. Mostly used for inheritance below."""

    interval_days: Union[int, float] = Field(
        description="""How far to look in the given direction (ahead for outcomes,
            behind for predictors)""",
    )

    resolve_multiple_fn: Union[Callable, str] = Field(
        description="""A function used for resolving multiple values within the
            interval_days.""",
    )

    key_for_resolve_multiple: Optional[str] = Field(
        default=None,
        description="""Key used to lookup the resolve_multiple_fn in the
            resolve_multiple_fns registry. Used for column name generation. Only
            required if you don't specify a resolve_multiple_fn. Call
            timeseriesflattener.resolve_multiple_fns.resolve_multiple_fns.get_all()
            for a list of options.""",
    )

    fallback: Union[Callable, int, float, str] = Field(
        description="""Which value to use if no values are found within interval_days.""",
    )

    allowed_nan_value_prop: float = Field(
        default=0.0,
        description="""If NaN is higher than this in the input dataframe during
            resolution, raise an error.""",
    )

    entity_id_col_name: str = Field(
        default="entity_id",
        description="""Col name for ids in the input dataframe.""",
    )

    loader_kwargs: Optional[dict] = Field(
        default=None,
        description="""Optional kwargs passed onto the data loader.""",
    )

    def __init__(self, **data: Any):
        if not hasattr(self, "key_for_resolve_multiple") and callable(
            data["resolve_multiple_fn"],
        ):
            data["key_for_resolve_multiple"] = data["resolve_multiple_fn"].__name__

        # Convert resolve_multiple_str to fn and add appropriate name
        if isinstance(data["resolve_multiple_fn"], str):
            data["key_for_resolve_multiple"] = data["resolve_multiple_fn"]

            data["resolve_multiple_fn"] = resolve_multiple_fns.get_all()[
                data["resolve_multiple_fn"]
            ]

        super().__init__(**data)

        self.resolve_multiple_fn = data["resolve_multiple_fn"]

        # override fallback strings with objects
        if self.fallback == "nan":
            self.fallback = float("nan")

    def get_col_str(self, additional_feature_name: Optional[str] = None) -> str:
        """Generate the column name for the output column.

        Args:
            additional_feature_name (Optional[str]): additional feature name to
                append to the column name.
        """
        feature_name = self.feature_name
        if additional_feature_name:
            feature_name = feature_name + "-" + str(additional_feature_name)
        col_str = f"{self.prefix}_{feature_name}_within_{self.interval_days}_days_{self.key_for_resolve_multiple}_fallback_{self.fallback}"
        return col_str


class PredictorSpec(TemporalSpec):
    """Specification for a single predictor, where the df has been resolved.

    Fields:
        values_loader (Optional[Callable]):
            Loader for the df. Tries to resolve from the data_loaders registry,
            then calls the function which should return a dataframe.
        values_name (Optional[str]):
            A string that maps to a key in a dictionary instantiated by
            `split_df_and_register_to_dict`. Each key corresponds to a dataframe, which
            is a subset of the df where the values_name == key.
        loader_kwargs (Optional[dict]):
            Optional kwargs passed onto the data loader.
        values_df (Optional[DataFrame]):
            Dataframe with the values.
        feature_name (str):
            The name of the feature. Used for column name generation, e.g.
            <prefix>_<feature_name>.
        prefix (str):
            The prefix used for column name generation, e.g.
            <prefix>_<feature_name>. Defaults to: pred.
        input_col_name_override (Optional[str]):
            An override for the input column name. If None, will  attempt
            to infer it by looking for the only column that doesn't match id_col_name
            or timestamp_col_name.
        output_col_name_override (Optional[str]):
            Override the generated column name after flattening the time series
        interval_days (Union[int, float]):
            How far to look in the given direction (ahead for outcomes,
            behind for predictors)
        resolve_multiple_fn (Union[Callable, str]):
            A function used for resolving multiple values within the
            interval_days.
        key_for_resolve_multiple (Optional[str]):
            Key used to lookup the resolve_multiple_fn in the
            resolve_multiple_fns registry. Used for column name generation. Only
            required if you don't specify a resolve_multiple_fn. Call
            timeseriesflattener.resolve_multiple_fns.resolve_multiple_fns.get_all()
            for a list of options.
        fallback (Union[Callable, int, float, str]):
            Which value to use if no values are found within interval_days.
        allowed_nan_value_prop (float):
            If NaN is higher than this in the input dataframe during
            resolution, raise an error. Defaults to: 0.0.
        entity_id_col_name (str):
            Col name for ids in the input dataframe. Defaults to: entity_id.
        lookbehind_days (Union[int, float]):
            How far behind to look for values
    """

    class Doc:
        short_description = (
            """Specification for a single predictor, where the df has been resolved."""
        )

    prefix: str = Field(
        default="pred",
        description="""The prefix used for column name generation, e.g.
            <prefix>_<feature_name>.""",
    )

    lookbehind_days: Union[int, float] = Field(
        description="""How far behind to look for values""",
    )

    def __init__(self, **data: Any):
        if "lookbehind_days" in data:
            data["interval_days"] = data["lookbehind_days"]

        data["lookbehind_days"] = data["interval_days"]

        if not data["interval_days"] and not data["lookbehind_days"]:
            raise ValueError("lookbehind_days or interval_days must be specified.")

        super().__init__(**data)


class TextPredictorSpec(PredictorSpec):
    """Specification for a text predictor, where the df has been resolved.

    Fields:
        values_loader (Optional[Callable]):
            Loader for the df. Tries to resolve from the data_loaders registry,
            then calls the function which should return a dataframe.
        values_name (Optional[str]):
            A string that maps to a key in a dictionary instantiated by
            `split_df_and_register_to_dict`. Each key corresponds to a dataframe, which
            is a subset of the df where the values_name == key.
        loader_kwargs (Optional[dict]):
            Optional kwargs passed onto the data loader.
        values_df (Optional[DataFrame]):
            Dataframe with the values.
        feature_name (str):
            The name of the feature. Used for column name generation, e.g.
            <prefix>_<feature_name>.
        prefix (str):
            The prefix used for column name generation, e.g.
            <prefix>_<feature_name>. Defaults to: pred.
        input_col_name_override (Optional[str]):
            An override for the input column name. If None, will  attempt
            to infer it by looking for the only column that doesn't match id_col_name
            or timestamp_col_name.
        output_col_name_override (Optional[str]):
            Override the generated column name after flattening the time series
        interval_days (Union[int, float]):
            How far to look in the given direction (ahead for outcomes,
            behind for predictors)
        resolve_multiple_fn (Union[Callable, str]):
            A function used for resolving multiple values within the
        interval_days, i.e. how to combine texts within the lookbehind window.
        Defaults to: 'concatenate'. Other possible options are 'latest' and
        'earliest'. Defaults to: concatenate.
        key_for_resolve_multiple (Optional[str]):
            Key used to lookup the resolve_multiple_fn in the
            resolve_multiple_fns registry. Used for column name generation. Only
            required if you don't specify a resolve_multiple_fn. Call
            timeseriesflattener.resolve_multiple_fns.resolve_multiple_fns.get_all()
            for a list of options.
        fallback (Union[Callable, int, float, str]):
            Which value to use if no values are found within interval_days.
        allowed_nan_value_prop (float):
            If NaN is higher than this in the input dataframe during
            resolution, raise an error. Defaults to: 0.0.
        entity_id_col_name (str):
            Col name for ids in the input dataframe. Defaults to: entity_id.
        lookbehind_days (Union[int, float]):
            How far behind to look for values
        embedding_fn (Callable):
            A function used for embedding the text. Should take a
        pandas series of strings and return a pandas dataframe of embeddings.
        Defaults to: None.
        embedding_fn_kwargs (Optional[dict]):
            Optional kwargs passed onto the embedding_fn."""

    class Doc:
        short_description = (
            """Specification for a text predictor, where the df has been resolved."""
        )

    embedding_fn: Callable = Field(
        description="""A function used for embedding the text. Should take a
        pandas series of strings and return a pandas dataframe of embeddings.
        Defaults to: None.""",
    )
    embedding_fn_kwargs: Optional[dict] = Field(
        default=None,
        description="""Optional kwargs passed onto the embedding_fn.""",
    )
    resolve_multiple_fn: Union[Callable, str] = Field(
        default="concatenate",
        description="""A function used for resolving multiple values within the
        interval_days, i.e. how to combine texts within the lookbehind window.
        Defaults to: 'concatenate'. Other possible options are 'latest' and
        'earliest'.""",
    )


class OutcomeSpec(TemporalSpec):
    """Specification for a single outcome, where the df has been resolved.

    Fields:
        values_loader (Optional[Callable]):
            Loader for the df. Tries to resolve from the data_loaders registry,
            then calls the function which should return a dataframe.
        values_name (Optional[str]):
            A string that maps to a key in a dictionary instantiated by
            `split_df_and_register_to_dict`. Each key corresponds to a dataframe, which
            is a subset of the df where the values_name == key.
        loader_kwargs (Optional[dict]):
            Optional kwargs passed onto the data loader.
        values_df (Optional[DataFrame]):
            Dataframe with the values.
        feature_name (str):
            The name of the feature. Used for column name generation, e.g.
            <prefix>_<feature_name>.
        prefix (str):
            The prefix used for column name generation, e.g.
            <prefix>_<outcome_name>. Defaults to: outc.
        input_col_name_override (Optional[str]):
            An override for the input column name. If None, will  attempt
            to infer it by looking for the only column that doesn't match id_col_name
            or timestamp_col_name.
        output_col_name_override (Optional[str]):
            Override the generated column name after flattening the time series
        interval_days (Union[int, float]):
            How far to look in the given direction (ahead for outcomes,
            behind for predictors)
        resolve_multiple_fn (Union[Callable, str]):
            A function used for resolving multiple values within the
            interval_days.
        key_for_resolve_multiple (Optional[str]):
            Key used to lookup the resolve_multiple_fn in the
            resolve_multiple_fns registry. Used for column name generation. Only
            required if you don't specify a resolve_multiple_fn. Call
            timeseriesflattener.resolve_multiple_fns.resolve_multiple_fns.get_all()
            for a list of options.
        fallback (Union[Callable, int, float, str]):
            Which value to use if no values are found within interval_days.
        allowed_nan_value_prop (float):
            If NaN is higher than this in the input dataframe during
            resolution, raise an error. Defaults to: 0.0.
        entity_id_col_name (str):
            Col name for ids in the input dataframe. Defaults to: entity_id.
        incident (bool):
            Whether the outcome is incident or not.
            I.e., incident outcomes are outcomes you can only experience once.
            For example, type 2 diabetes is incident. Incident outcomes can be handled
            in a vectorised way during resolution, which is faster than non-incident outcomes.
        lookahead_days (Union[int, float]):
            How far ahead to look for values
    """

    class Doc:
        short_description = (
            """Specification for a single outcome, where the df has been resolved."""
        )

    prefix: str = Field(
        default="outc",
        description="""The prefix used for column name generation, e.g.
            <prefix>_<outcome_name>.""",
    )

    incident: bool = Field(
        description="""Whether the outcome is incident or not.
            I.e., incident outcomes are outcomes you can only experience once.
            For example, type 2 diabetes is incident. Incident outcomes can be handled
            in a vectorised way during resolution, which is faster than non-incident outcomes.""",
    )

    lookahead_days: Union[int, float] = Field(
        description="""How far ahead to look for values""",
    )

    def __init__(self, **data: Any):
        if "lookahead_days" in data:
            data["interval_days"] = data["lookahead_days"]

        data["lookahead_days"] = data["interval_days"]

        super().__init__(**data)

    def get_col_str(self, additional_feature_name: Optional[str] = None) -> str:
        """Get the column name for the output column."""
        col_str = super().get_col_str(additional_feature_name=additional_feature_name)

        if self.is_dichotomous():
            col_str += "_dichotomous"

        return col_str

    def is_dichotomous(self) -> bool:
        """Check if the outcome is dichotomous."""
        col_name = (
            "value"
            if not self.input_col_name_override
            else self.input_col_name_override
        )

        return len(self.values_df[col_name].unique()) <= 2  # type: ignore


class _MinGroupSpec(BaseModel):
    class Doc:
        short_description = """Minimum specification for a group of features,
        whether they're looking ahead or behind.

        Used to generate combinations of features."""

    values_loader: Optional[List[str]] = Field(
        default=None,
        description="""Loader for the df. Tries to resolve from the data_loaders
            registry, then calls the function which should return a dataframe.""",
    )

    values_name: Optional[List[str]] = Field(
        default=None,
        description="""List of strings that corresponds to a key in a dictionary
            of multiple dataframes that correspods to a name of a type of values.""",
    )

    values_df: Optional[pd.DataFrame] = Field(
        default=None,
        description="""Dataframe with the values.""",
    )

    input_col_name_override: Optional[str] = Field(
        default=None,
        description="""Override for the column name to use as values in df.""",
    )

    output_col_name_override: Optional[str] = Field(
        default=None,
        description="""Override for the column name to use as values in the
            output df.""",
    )

    resolve_multiple_fn: List[Union[Callable, str]] = Field(
        description="""Name of resolve multiple fn, resolved from
            resolve_multiple_functions.py""",
    )

    fallback: List[Union[Callable, str]] = Field(
        description="""Which value to use if no values are found within interval_days.""",
    )

    allowed_nan_value_prop: List[float] = Field(
        default=[0.0],
        description="""If NaN is higher than this in the input dataframe during
            resolution, raise an error.""",
    )

    prefix: Optional[str] = Field(
        default=None,
        description="""Prefix for column name, e.g. <prefix>_<feature_name>.""",
    )

    loader_kwargs: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="""Optional kwargs for the values_loader.""",
    )

    def _check_loaders_are_valid(self):
        """Check that all loaders can be resolved from the data_loaders catalogue."""
        invalid_loaders = list(
            set(self.values_loader) - set(data_loaders.get_all().keys()),
        )
        if len(invalid_loaders) != 0:
            # New line variable as f-string can't handle backslashes
            nl = "\n"  # pylint: disable = invalid-name
            available_loaders = [str(loader) for loader in data_loaders.get_all()]

            avail_loaders_str = nl.join(available_loaders)

            if len(available_loaders) == 0:
                avail_loaders_str = "No loaders available."

            raise ValueError(
                f"""Some loader strings could not be resolved in the data_loaders catalogue. Did you make a typo? If you want to add your own loaders to the catalogue, see explosion / catalogue on GitHub for info.
                {nl*2}Loaders that could not be resolved:"""
                f"""{nl}{nl.join(str(loader) for loader in invalid_loaders)}{nl}{nl}"""
                f"""Available loaders:{nl}{avail_loaders_str}""",
            )

    # Prefix for the column name. Overrides the default prefix for the feature type.

    def __init__(self, **data: Any):
        super().__init__(**data)

        # Check that all passed loaders are valid
        if self.values_loader is not None:
            self._check_loaders_are_valid()

        if self.output_col_name_override:
            input_col_name = (
                "value"
                if not self.input_col_name_override
                else self.input_col_name_override
            )

            self.values_df.rename(  # type: ignore
                columns={input_col_name: self.output_col_name_override},
                inplace=True,  # noqa
            )


def create_feature_combinations_from_dict(
    d: Dict[str, Union[str, list]],
) -> List[Dict[str, Union[str, float, int]]]:
    """Create feature combinations from a dictionary of feature specifications.

    Only unpacks the top level of lists.

    Args:
        d (Dict[str]): A dictionary of feature specifications.

    Returns
    -------
        List[Dict[str]]: list of all possible combinations of the arguments.
    """
    # Make all elements iterable
    d = {k: v if isinstance(v, (list, tuple)) else [v] for k, v in d.items()}
    keys, values = zip(*d.items())

    # Create all combinations of top level elements
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

    return permutations_dicts


def create_specs_from_group(
    feature_group_spec: _MinGroupSpec,
    output_class: _AnySpec,
) -> List[_AnySpec]:
    """Create a list of specs from a GroupSpec."""
    # Create all combinations of top level elements
    # For each attribute in the FeatureGroupSpec

    feature_group_spec_dict = feature_group_spec.__dict__

    permuted_dicts = create_feature_combinations_from_dict(d=feature_group_spec_dict)

    return [output_class(**d) for d in permuted_dicts]  # type: ignore


class PredictorGroupSpec(_MinGroupSpec):
    """Specification for a group of predictors.

    Fields:
        values_loader (Optional[List[str]]):
            Loader for the df. Tries to resolve from the data_loaders
            registry, then calls the function which should return a dataframe.
        values_name (Optional[List[str]]):
            List of strings that corresponds to a key in a dictionary
            of multiple dataframes that correspods to a name of a type of values.
        values_df (Optional[DataFrame]):
            Dataframe with the values.
        input_col_name_override (Optional[str]):
            Override for the column name to use as values in df.
        output_col_name_override (Optional[str]):
            Override for the column name to use as values in the
            output df.
        resolve_multiple_fn (List[Union[Callable, str]]):
            Name of resolve multiple fn, resolved from
            resolve_multiple_functions.py
        fallback (List[Union[Callable, str]]):
            Which value to use if no values are found within interval_days.
        allowed_nan_value_prop (List[float]):
            If NaN is higher than this in the input dataframe during
            resolution, raise an error. Defaults to: [0.0].
        prefix (str):
            Prefix for column name, e,g, <prefix>_<feature_name>. Defaults to: pred.
        loader_kwargs (Optional[List[Dict[str, Any]]]):
            Optional kwargs for the values_loader.
        lookbehind_days (List[Union[int, float]]):
            How far behind to look for values
    """

    class Doc:
        short_description = """Specification for a group of predictors."""

    prefix: str = Field(
        default="pred",
        description="""Prefix for column name, e,g, <prefix>_<feature_name>.""",
    )

    lookbehind_days: List[Union[int, float]] = Field(
        description="""How far behind to look for values""",
    )

    def create_combinations(self) -> List[PredictorSpec]:
        """Create all combinations from the group spec."""
        return create_specs_from_group(  # type: ignore
            feature_group_spec=self,
            output_class=PredictorSpec,  # type: ignore
        )


class OutcomeGroupSpec(_MinGroupSpec):
    """Specification for a group of outcomes.

    Fields:
    values_loader (Optional[List[str]]):
        Loader for the df. Tries to resolve from the data_loaders
        registry, then calls the function which should return a dataframe.
    values_name (Optional[List[str]]):
        List of strings that corresponds to a key in a dictionary
        of multiple dataframes that correspods to a name of a type of values.
    values_df (Optional[DataFrame]):
        Dataframe with the values.
    input_col_name_override (Optional[str]):
        Override for the column name to use as values in df.
    output_col_name_override (Optional[str]):
        Override for the column name to use as values in the
        output df.
    resolve_multiple_fn (List[Union[Callable, str]]):
        Name of resolve multiple fn, resolved from
        resolve_multiple_functions.py
    fallback (List[Union[Callable, str]]):
        Which value to use if no values are found within interval_days.
    allowed_nan_value_prop (List[float]):
        If NaN is higher than this in the input dataframe during
        resolution, raise an error. Defaults to: [0.0].
    prefix (str):
        Prefix for column name, e.g. <prefix>_<feature_name>. Defaults to: outc.
    loader_kwargs (Optional[List[Dict[str, Any]]]):
        Optional kwargs for the values_loader.
    incident (Sequence[bool]):
        Whether the outcome is incident or not, i.e. whether you
        can experience it more than once. For example, type 2 diabetes is incident.
        Incident outcomes can be handled in a vectorised way during resolution,
         which is faster than non-incident outcomes.
    lookahead_days (List[Union[int, float]]):
        How far ahead to look for values
    """

    class Doc:
        short_description = """Specification for a group of outcomes."""

    prefix: str = Field(
        default="outc",
        description="""Prefix for column name, e.g. <prefix>_<feature_name>.""",
    )

    incident: Sequence[bool] = Field(
        description="""Whether the outcome is incident or not, i.e. whether you
            can experience it more than once. For example, type 2 diabetes is incident.
            Incident outcomes can be handled in a vectorised way during resolution,
             which is faster than non-incident outcomes.""",
    )

    lookahead_days: List[Union[int, float]] = Field(
        description="""How far ahead to look for values""",
    )

    def create_combinations(self) -> List[OutcomeSpec]:
        """Create all combinations from the group spec."""
        return create_specs_from_group(  # type: ignore
            feature_group_spec=self,
            output_class=OutcomeSpec,  # type: ignore
        )
