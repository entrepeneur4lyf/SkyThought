import copy
from typing import List

from skythought.evals.util.common import has_code

from ..base import Scorer
from .livecodebench_util import (
    has_test_type,
    post_process_code,
    unsafe_lcb_runTests_ray,
)


class LiveCodeBenchScorer(Scorer):

    TIMEOUT = 6

    def score(self, row: dict) -> bool:
        row = map_to_example(row)

        code_filter_result = has_code(row["response"])
        last_code = None
        if len(code_filter_result) == 0:
            return False
        else:
            last_code = code_filter_result[-1]
            problem_to_check = copy.deepcopy(row)

        result_list = unsafe_lcb_runTests_ray(
            problem_to_check,
            post_process_code(last_code),
            self.TIMEOUT,
            runtime_debug=False,
            is_extracted=row["is_stdin"],
        )
        details = [r[0] for r in result_list]
        all_passed = all(details)

        result = ""
        if result_list and all_passed:
            result = "passed"

        return result == "passed"

    @property
    def expected_keys(self) -> List[str]:
        return [
            "question_content",
            "private_test_cases",
            "public_test_cases",
            "difficulty",
            "question_id",
            "starter_code",
        ]


def map_to_example(row):
    return {
        "prompt": row["question_content"],
        "test": row["private_test_cases"],
        "entry_point": row["starter_code"],
        "canonical_solution": "",  # seems like live code bench lite does not have this field
        "task_id": row["question_id"],
        "is_stdin": has_test_type(row["public_test_cases"], "stdin"),
        "public_test_cases": row["public_test_cases"],
        "difficulty": row["difficulty"],
    }
