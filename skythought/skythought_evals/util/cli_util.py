from ast import literal_eval

import typer
from click.core import ParameterSource


def parse_list_of_args(vals: str) -> dict:
    vals = vals.replace(" ", "")
    if not len(vals):
        return {}
    return {k: literal_eval(v) for k, v in [val.split("=") for val in vals.split(",")]}


def get_user_provided_params(ctx: typer.Context) -> dict:
    # click_ctx: click.Context = ctx.get_click_context()
    # Initialize a dictionary to hold user-provided parameters
    user_provided_params = {}

    # Iterate over all parameters of the command
    for param in ctx.command.params:
        # Get the value of the parameter
        value = ctx.params.get(param.name)

        # Determine the source of the parameter's value
        source = ctx.get_parameter_source(param)

        # Check if the parameter was set via the command line
        if source == ParameterSource.COMMANDLINE:
            user_provided_params[param.name] = value
    return user_provided_params
