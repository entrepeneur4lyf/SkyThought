from typing import Any, AsyncIterator, Callable, Dict, List

from ray.data.llm import Processor
from ray.llm._internal.batch.stages.base import StatefulStageUDF

from ..base import Scorer


class ScoreUDF(StatefulStageUDF):
    def __init__(self, scorer: Scorer):
        self.scorer = scorer

    async def udf(self, batch: List[Dict[str, Any]]) -> AsyncIterator[Dict[str, Any]]:
        for row in batch:
            result = self.scorer(row)
            yield {
                self.IDX_IN_BATCH_COLUMN: row[self.IDX_IN_BATCH_COLUMN],
                "score": result,
            }


def build_verifiable_score_processor(
    scorer: Scorer,
    preprocess: Callable[[Dict[str, Any]], Dict[str, Any]],
    postprocess: Callable[[Dict[str, Any]], Dict[str, Any]],
) -> Processor:
    return Processor(
        preprocess=preprocess, stages=[ScoreUDF(scorer)], postprocess=postprocess
    )
