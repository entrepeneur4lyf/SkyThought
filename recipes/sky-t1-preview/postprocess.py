import copy
import json
from typing import Any, Dict

import numpy as np
import ray

from skythought.evals.scoring.base import Scorer
from skythought.evals.tasks.apps.apps_util import run_test as apps_run_test
from skythought.evals.util.common import has_code

STILL2_SYSTEM_PROMPT = "Your role as an assistant involves thoroughly exploring questions through a systematic long \
thinking process before providing the final precise and accurate solutions. This requires \
engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, \
backtracing, and iteration to develop well-considered thinking process. \
Please structure your response into two main sections: Thought and Solution. \
In the Thought section, detail your reasoning process using the specified format: \
<|begin_of_thought|> {thought with steps separated with '\n\n'} \
<|end_of_thought|> \
Each step should include detailed considerations such as analisying questions, summarizing \
relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining \
any errors, and revisiting previous steps. \
In the Solution section, based on various attempts, explorations, and reflections from the Thought \
section, systematically present the final solution that you deem correct. The solution should \
remain a logical, accurate, concise expression style and detail necessary step needed to reach the \
conclusion, formatted as follows: \
<|begin_of_solution|> \
{final formatted, precise, and clear solution} \
<|end_of_solution|> \
Now, try to solve the following question through the above guidelines:"


class APPSScorer(Scorer):
    def score(self, row: Dict[str, Any]):
        TIMEOUT = 10
        code_filter_result = has_code(row["response"])
        if len(code_filter_result) == 0:
            return False
        else:
            last_code = code_filter_result[-1]
            problem_to_check = copy.deepcopy(row)
            problem_to_check["input_output"] = json.loads(row["input_output"])
            try:
                problem_to_check["solutions"] = json.loads(row["solutions"])
            except Exception:
                problem_to_check["solutions"] = ""

        @ray.remote
        def _temp_run(problem, generation, debug):
            try:
                result = apps_run_test(problem=problem, test=generation, debug=debug)
                return result
            except Exception:
                pass

        result = ray.get(
            _temp_run.remote(problem_to_check, last_code, False), timeout=TIMEOUT + 1
        )

        return bool(result and np.all(result[0]))


class TACOScorer(Scorer):
    def score(self, row: Dict[str, Any]):
        return True


def convert_to_sharegpt_format(row: Dict[str, Any]):
    prompt = row["user_input"]
    # accept
    # Create the conversation format
    conversations = [
        {"from": "user", "value": prompt},
        {
            "from": "assistant",
            "value": row["formatted_response"],
        },
    ]

    # Prepare the final structure
    cur_data = {
        "system": STILL2_SYSTEM_PROMPT,
        "conversations": conversations,
    }

    return cur_data
