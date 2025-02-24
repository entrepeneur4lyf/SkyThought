from ..util.math_parsing_util import extract_answer, math_equal
from .base import Scorer

try:
    from math_verify import parse as mv_parse
    from math_verify import verify as mv_verify
except ImportError:
    mv_parse = None
    mv_verify = None


class MathEqualScorer(Scorer):
    """Scorer for math based on the `math_equal` function from Qwen Math"""

    def __init__(self, response_key: str, answer_key: str):
        self.response_key = response_key
        self.answer_key = answer_key

    def __call__(self, row: dict) -> bool:
        try:
            pred = extract_answer(row[self.response_key])
            ref = extract_answer(row[self.answer_key])
        except Exception:
            return False
        return math_equal(pred, ref)


class MathVerifyScorer(Scorer):
    """
    Scorer for math based on the `math_verify` function from HuggingFace
    """

    def __init__(self, response_key: str, answer_key: str):

        self.response_key = response_key
        self.answer_key = answer_key
        if mv_parse is None or mv_verify is None:
            raise ImportError(
                "math_verify is not installed. Please install it with `pip install math_verify`."
            )

    def __call__(self, row: dict) -> bool:
        try:
            pred = mv_parse(row[self.response_key])
            ref = mv_parse(row[self.answer_key])
        except Exception:
            return False
        return mv_verify(pred, ref)
