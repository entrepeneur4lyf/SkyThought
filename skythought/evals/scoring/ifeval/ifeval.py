from typing import List

from skythought.evals.scoring.ifeval.instructions_main import (
    InputExample,
    test_instruction_following_loose,
    test_instruction_following_strict,
)

from ..base import Scorer


def process_results(doc, response):
    inp = InputExample(
        key=doc["key"],
        instruction_id_list=doc["instruction_id_list"],
        prompt=doc["prompt"],
        kwargs=doc["kwargs"],
    )

    out_strict = test_instruction_following_strict(inp, response)
    out_loose = test_instruction_following_loose(inp, response)

    return {
        "prompt_level_strict_acc": out_strict.follow_all_instructions,
        "inst_level_strict_acc": out_strict.follow_instruction_list,
        "prompt_level_loose_acc": out_loose.follow_all_instructions,
        "inst_level_loose_acc": out_loose.follow_instruction_list,
    }


class IfEvalScorer(Scorer):
    """
    Scorer for the IF-Eval dataset. Requires the dataset to be in the format as https://huggingface.co/datasets/google/IFEval
    """

    def __call__(self, row: dict) -> bool:
        return process_results(row, row["response"])

    @property
    def expected_keys(self) -> List[str]:
        return ["instruction_id_list", "prompt", "kwargs", "key"]
