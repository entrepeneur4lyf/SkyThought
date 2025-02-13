import copy
from dataclasses import dataclass
from enum import Enum
from importlib import resources
from typing import Literal, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field
from vllm import SamplingParams as VLLMSamplingParams

TEMPERATURE_DEFAULT = 0
TOP_P_DEFAULT = 1
MAX_TOKENS_DEFAULT = 32768


class Backend(str, Enum):
    VLLM = "vllm"
    OPENAI = "openai"
    RAY = "ray"


class OpenAISamplingParams(BaseModel):
    temperature: float = TEMPERATURE_DEFAULT
    top_p: float = TOP_P_DEFAULT
    n: int = 1
    max_tokens: int = MAX_TOKENS_DEFAULT
    reasoning_effort: Optional[float] = None
    frequency_penalty: Optional[float] = None


class SamplingParameters(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    params: Union[OpenAISamplingParams, VLLMSamplingParams]

    @classmethod
    def from_dict(cls, backend: Backend, params: dict):
        params = copy.deepcopy(params)
        if backend == Backend.OPENAI:
            return cls(params=OpenAISamplingParams(**params))
        # Currently, ray-data based processor only supports vllm as the inference engine
        elif backend in [Backend.VLLM, Backend.RAY]:
            return cls(params=VLLMSamplingParams(**params))
        else:
            raise ValueError(f"Invalid backend type: {backend}")

    def __repr__(self):
        return f"SamplingParameters(params={self.params})"

    def to_dict(self):
        if isinstance(self.params, OpenAISamplingParams):
            return self.params.model_dump()
        elif isinstance(self.params, VLLMSamplingParams):
            return {k: getattr(self.params, k) for k in self.params.__annotations__}
        else:
            raise ValueError(f"Invalid sampling parameters type: {type(self.params)}")


class OpenAIClientArgs(BaseModel):
    api_key: Optional[str] = Field(None, description="OpenAI API key")
    base_url: Optional[str] = Field(None, description="OpenAI base URL")
    project: Optional[str] = Field(None, description="OpenAI project")
    organization: Optional[str] = Field(None, description="OpenAI organization")


class RayLLMEngineArgs(BaseModel):

    tp: Optional[int] = Field(description="Tensor parallelism size")
    num_replicas: Optional[int] = Field(description="Number of replicas to use for Ray")
    batch_size: Optional[int] = Field(description="Batch size for Ray")
    gpu_memory_utilization: Optional[float] = Field(
        description="GPU memory utilization for the vLLM engine"
    )
    dtype: Literal["float32", "float16", "bfloat16", "float8"] = Field(
        "float32", description="Data type for inference engine."
    )

    def get_ray_llm_config(self):
        with resources.open_text(
            "skythought_evals.ray_configs", "ray_config.yaml"
        ) as f:
            default_config = yaml.safe_load(f)

        if self.tp is not None:
            default_config["engine_kwargs"]["tensor_parallel_size"] = self.tp

        if self.num_replicas is not None:
            default_config["env_config"]["num_replicas"] = self.num_replicas

        if self.batch_size is not None:
            default_config["env_config"]["batch_size"] = self.batch_size

        if self.gpu_memory_utilization is not None:
            default_config["engine_kwargs"][
                "gpu_memory_utilization"
            ] = self.gpu_memory_utilization

        # FIXME (sumanthrh): there can be a corner case when we support providing a config yaml directly, and this will override the dtype
        default_config["engine_kwargs"]["dtype"] = self.dtype

        return default_config


@dataclass
class BackendParameters:
    model_config = ConfigDict(arbitrary_types_allowed=True)

    params: Union[dict, OpenAIClientArgs, RayLLMEngineArgs]

    @classmethod
    def from_dict(cls, backend_type: str, params: dict):
        if backend_type == "rayllm":
            return cls(params=RayLLMEngineArgs(**params))
        elif backend_type == "vllm":
            # passed directly to LLM(..) instantiation
            return cls(params=params)
        else:
            raise ValueError(f"Invalid backend type: {backend_type}")

    def to_dict(self):
        if isinstance(self.params, RayLLMEngineArgs):
            return self.params.model_dump()
        elif isinstance(self.params, dict):
            return self.params
        elif isinstance(self.params, OpenAIClientArgs):
            return self.params.model_dump()
        else:
            raise ValueError(f"Invalid backend parameters type: {type(self.params)}")
