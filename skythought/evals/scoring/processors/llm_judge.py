from abc import ABC, abstractmethod
from typing import Any, Dict

import ray.data
from ray.data.llm import Processor


class ScoreProcessor(ABC, Processor):

    @abstractmethod
    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        pass

    @abstractmethod
    def postprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        pass

    def __call_(self, ds: ray.data.Dataset):
        ds = ds.map(self.preprocess)
        ds = super().__call__(ds)
        ds = ds.map(self.postprocess)
