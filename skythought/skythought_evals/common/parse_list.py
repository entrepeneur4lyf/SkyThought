from ast import literal_eval


def parse_list_of_args(vals: str) -> dict:
    vals = vals.replace(" ", "")
    return {k: literal_eval(v) for k, v in [val.split("=") for val in vals.split(",")]}
