import logging
import os
from pathlib import Path

import typer
from skythought_evals.common.entities import (
    Backend,
    BackendParameters,
    SamplingParameters,
)
from skythought_evals.models import ModelConfig
from skythought_evals.tasks import TASK_HANDLER_MAP, TASK_NAMES_TO_YAML, TaskConfig
from skythought_evals.util.cli_util import parse_list_of_args
from skythought_evals.util.common import set_seed
from typing_extensions import Annotated

from .inference_and_check import generate_and_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = typer.Typer(pretty_exceptions_enable=False)

TEMP_DEFAULT = typer.Option([0], help="Temperature for sampling.")


def to_tuple(d) -> tuple:
    if isinstance(d, dict):
        return tuple(map(to_tuple, d.items()))
    elif isinstance(d, (set, list, tuple)):
        return tuple(map(to_tuple, d))
    else:
        return d


def get_output_dir(
    result_dir,
    *,
    model_config: ModelConfig,
    task: str,
    task_config: TaskConfig,
    start: int,
    end: int,
    sampling_params: SamplingParameters,
    backend: Backend,
    backend_params: BackendParameters,
) -> Path:
    # name -> model_config.name_task_{parameter_hash}
    # ge ta single has for all the remaining parameters - backend, backend_args, sampling_params
    parameter_hash = hash(
        (
            to_tuple(task_config.model_dump()),
            backend,
            to_tuple(backend_params.to_dict()),
            to_tuple(sampling_params.to_dict()),
        )
    )
    parameter_hash = abs(parameter_hash) % 10**6  # just 6 digits

    return (
        Path(result_dir)
        / f"{model_config.model_id.replace('/', '_')}_{task}_s{start}_e{end}_{parameter_hash}"
    )


@app.command()
def evaluate(
    ctx: typer.Context,
    task: Annotated[
        str, typer.Option(..., help="Task to process.", case_sensitive=False)
    ],
    model: Annotated[str, typer.Option(..., help="The model to run")],
    # tp: int = typer.Option(8, help="Tensor Parallelism Degree"),
    # max_tokens: int = typer.Option(32768, help="Max tokens for the model."),
    # subset: Annotated[str, typer.Option(None, help="Subset for the dataset.")],
    backend: Annotated[
        Backend,
        typer.Option(
            help="Backend to use for inference.",
            case_sensitive=False,
        ),
    ] = Backend.VLLM,
    backend_args: Annotated[
        str,
        typer.Option(
            help="Backend parameters to use for inference.",
            case_sensitive=False,
        ),
    ] = "",
    sampling_params: Annotated[
        str,
        typer.Option(
            help="Sampling parameters to use for inference.",
            case_sensitive=False,
        ),
    ] = "temperature=0,top_p=1,max_tokens=32768",
    result_dir: Annotated[
        str,
        typer.Option(
            help="Result dir to save files.",
        ),
    ] = None,
    system_prompt_name: Annotated[
        str, typer.Option(help="System prompt name to use")
    ] = None,
    system_prompt: Annotated[str, typer.Option(help="System prompt to use")] = None,
    n: Annotated[
        int, typer.Option(help="Number of samples generated per problem.")
    ] = None,
    seed: Annotated[int, typer.Option(help="Random seed.")] = 41,
    assistant_prefill: Annotated[
        str,
        typer.Option(help=r'Assistant prefill for the model response. Ex: "<think>\n"'),
    ] = None,
    as_test: Annotated[
        bool, typer.Option(help="Perform a test run on 10 samples of the dataset.")
    ] = False,
    overwrite: Annotated[
        bool, typer.Option(help="Overwrite existing results.")
    ] = False,
    batch_size: Annotated[
        int,
        typer.Option(
            help="Batch size for inference. only applicable for the vllm backend."
        ),
    ] = 64,
    dtype: str = typer.Option(
        "float32", help="dtype for inference with vLLM.", case_sensitive=False
    ),
):
    # # load ray config
    # if use_ray:
    #     warnings.warn(
    #         "`tp` CLI argument is not compatible with `use-ray` and will be ignored. Please configure tensor parallel size in the `ray_config` YAML"
    #         " or override the value with the argument `ray-config-tensor-parallel-size` ",
    #         stacklevel=1,
    #     )
    #     if not ray_config:
    #         # load default
    #         ray_config = os.path.join(
    #             os.path.dirname(__file__), "ray_configs/ray_config.yaml"
    #         )
    set_seed(seed)

    if batch_size != 64 and backend != Backend.VLLM:
        raise ValueError("Batch size is only supported for the vllm backend.")

    # user_provided_params = get_user_provided_params(ctx)

    # enable hf_transfer if not overridden by the user
    if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", None) is None:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    if task not in TASK_NAMES_TO_YAML:
        raise ValueError(
            f"Task {task} not found. Should be one of {TASK_NAMES_TO_YAML.keys()}"
        )

    sampling_params_as_dict = parse_list_of_args(sampling_params)
    backend_args_as_dict = parse_list_of_args(backend_args)
    sampling_params: SamplingParameters = SamplingParameters.from_dict(
        backend, sampling_params_as_dict
    )
    backend_params: BackendParameters = BackendParameters.from_dict(
        backend, backend_args_as_dict
    )

    # set dtype
    if backend == Backend.RAY:
        backend_params.params.dtype = dtype
    elif backend == Backend.VLLM:
        backend_params.params["dtype"] = dtype

    if sampling_params.params.top_p < 1 and model.startswith("openai/o1"):
        print(
            "OpenAI o1 models do not support `top_p` sampling. Resetting `top_p` to 1"
        )
        sampling_params.params.top_p = 1

    logger.info(
        f"Temperature: {sampling_params.params.temperature}, top_p: {sampling_params.params.top_p}, max_tokens: {sampling_params.params.max_tokens}"
    )
    if n is not None:
        sampling_params.params.n = n

    if sampling_params.params.temperature == 0 and sampling_params.params.n > 1:
        sampling_params.params.n = 1
        logger.warning(
            "Warning: Temperature 0 does not support multiple samples. Setting n=1."
        )

    start = 0
    end = -1
    if as_test:
        start = 0
        end = 10
        sampling_params.params.max_tokens = 2048
        logger.info("Running test run with 10 samples and max tokens set to 2048.")

    task_config = TaskConfig.from_yaml(TASK_NAMES_TO_YAML[task])
    handler_name = task_config.handler
    handler_cls = TASK_HANDLER_MAP[handler_name]
    handler = handler_cls(task_config)

    model_config = ModelConfig.from_model_id(
        model, system_prompt_name, system_prompt, assistant_prefill
    )

    if result_dir is not None:
        output_dir = get_output_dir(
            result_dir,
            model_config=model_config,
            task=task,
            task_config=task_config,
            start=start,
            end=end,
            backend=backend,
            backend_params=backend_params,
            sampling_params=sampling_params,
        )
        if not overwrite and output_dir.exists():
            raise ValueError(
                f"Output directory {output_dir} already exists. pass `--overwrite` to overwrite."
            )
        # create result dir if not exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    run_configuration = {
        "task": {
            "name": task,
            "config": task_config.model_dump(),
            "start": start,
            "end": end,
        },
        "model": {
            "name": model,
            "config": model_config.model_dump(),
        },
        "backend": {
            "name": backend,
            "backend_args": backend_args_as_dict,
        },
        "sampling_params": sampling_params_as_dict,
    }

    # temperature_str = ",".join(map(str, temperatures))
    # file_suffix = (
    #     f"{model_config.name}_{task}_{split}_subset_{subset}_filter_{filter_difficulty}"
    #     + f"_s{start}_e{end}_t{temperature_str}_n{n}"
    # )
    # if (
    #     math_difficulty_lower_bound is not None
    #     or math_difficulty_upper_bound is not None
    # ):
    #     result_file = os.path.join(
    #         result_dir,
    #         f"{model_config.name}_{file_suffix}_{math_difficulty_upper_bound}.json",
    #     )
    # else:
    #     result_file = os.path.join(
    #         result_dir,
    #         f"{file_suffix}.json",
    #     )

    # FIXME
    generate_and_score(
        handler,
        model_config,
        backend,
        backend_params,
        sampling_params,
        output_dir,
        start,
        end,
        run_configuration,
        batch_size=batch_size,
    )

    # else:
    #     if use_ray:
    #         llm = None
    #     else:
    #         llm = (
    #             OpenAI()
    #             if model.startswith("openai")
    #             else LLM(model=model, tensor_parallel_size=tp, dtype=dtype)
    #         )
    #     if inference:
    #         perform_inference_and_save(
    #             handler, temperature, max_tokens, result_file, llm, model_config, args
    #         )
    #     else:
    #         perform_inference_and_check(
    #             handler, temperature, max_tokens, result_file, llm, model_config, args
    #         )


#  if check:
#         check if converted file exists
#         if (
#             math_difficulty_lower_bound is not None
#             or math_difficulty_upper_bound is not None
#         ):
#             converted_file = f"{result_dir}/converted_{file_suffix}.json"
#         else:
#             converted_file = f"{result_dir}/converted_{file_suffix}.json"
#         if os.path.exists(converted_file):
#             result_file = converted_file
#         perform_check(handler, temperature, result_file, args)
#         return


def generate(
    task: Annotated[
        str, typer.Option(..., help="Task to process.", case_sensitive=False)
    ],
    model: Annotated[str, typer.Option(..., help="The model to run")],
    backend: Annotated[
        Backend,
        typer.Option(
            help="Backend to use for inference.",
            case_sensitive=False,
        ),
    ] = Backend.VLLM,
    backend_args: Annotated[
        str,
        typer.Option(
            help="Backend parameters to use for inference.",
            case_sensitive=False,
        ),
    ] = "",
    sampling_params: Annotated[
        str,
        typer.Option(
            help="Sampling parameters to use for inference.",
            case_sensitive=False,
        ),
    ] = "temperature=0,top_p=1,max_tokens=32768",
    result_dir: Annotated[
        str,
        typer.Option(
            help="Result dir to save files.",
        ),
    ] = None,
    system_prompt_name: Annotated[
        str, typer.Option(help="System prompt name to use")
    ] = None,
    system_prompt: Annotated[str, typer.Option(help="System prompt to use")] = None,
    n: Annotated[
        int, typer.Option(help="Number of samples generated per problem.")
    ] = None,
    seed: Annotated[int, typer.Option(help="Random seed.")] = 41,
    assistant_prefill: Annotated[
        str,
        typer.Option(help=r'Assistant prefill for the model response. Ex: "<think>\n"'),
    ] = None,
    as_test: Annotated[
        bool, typer.Option(help="Perform a test run on 10 samples of the dataset.")
    ] = False,
    overwrite: Annotated[
        bool, typer.Option(help="Overwrite existing results.")
    ] = False,
    batch_size: Annotated[
        int,
        typer.Option(
            help="Batch size for inference. only applicable for the vllm backend."
        ),
    ] = 64,
    dtype: str = typer.Option(
        "float32", help="dtype for inference with vLLM.", case_sensitive=False
    ),
):
    pass


def score():
    pass


def main():
    app()
