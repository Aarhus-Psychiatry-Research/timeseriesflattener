from pydantic import BaseModel as PydanticBaseModel
from pydantic import Extra


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
    class Config:
        """Disallow  attributes not in the the class."""

        arbitrary_types_allowed = True
        allow_mutation = False
        extra = Extra.forbid
