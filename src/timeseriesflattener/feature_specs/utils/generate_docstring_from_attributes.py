from timeseriesflattener.utils.pydantic_basemodel import BaseModel


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
