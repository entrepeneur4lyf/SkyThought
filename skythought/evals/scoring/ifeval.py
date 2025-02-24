from .base import Scorer
from .utils.ifeval_utils import IF_FUNCTIONS_MAP

#  if isinstance(constraint, str):
#         constraint = json.loads(constraint)
#     if "func_name" not in constraint:
#         print("WARNING: constraint missing func_name")
#         print(constraint)
#         return False
#     # first, parse out the constraint string.
#     func_name = constraint.pop("func_name")
#     # get the function


class IfEvalScorer(Scorer):
    """
    Scorer for the IF-Eval dataset
    """

    def __init__(self, response_key: str, answer_key: str):
        self.response_key = response_key
        self.answer_key = answer_key

    def __call__(self, row: dict) -> bool:
        return IF_FUNCTIONS_MAP[row["task"]](
            row[self.response_key], row[self.answer_key]
        )
