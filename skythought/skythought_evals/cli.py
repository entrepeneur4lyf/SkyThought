import os
import warnings
from typing import List

import typer
from openai import OpenAI
from skythought_evals.inference_and_check import (
    perform_check,
    perform_inference_and_check,
    perform_inference_and_save,
)
from skythought_evals.models import ModelConfig
from skythought_evals.tasks import TASK_HANDLER_MAP, TASK_NAMES_TO_YAML, TaskConfig
from skythought_evals.util.common import set_seed
from vllm import LLM

app = typer.Typer()

TEMP_DEFAULT = typer.Option([0], help="Temperature for sampling.")


@app.command()
def main(
    task: str = typer.Option(..., help="Task to process.", case_sensitive=False),
    model: str = typer.Option(..., help="The model to run."),
    tp: int = typer.Option(8, help="Tensor Parallelism Degree"),
    max_tokens: int = typer.Option(32768, help="Max tokens for the model."),
    split: str = typer.Option(
        None, help="Split to use for the dataset (e.g., train, test)."
    ),
    subset: str = typer.Option(None, help="Subset for the dataset."),
    start: int = typer.Option(0, help="Start index."),
    end: int = typer.Option(-1, help="End index."),
    difficulty: str = typer.Option(
        None, help="Difficulty level. Example: 'easy', 'medium', 'hard'."
    ),
    filter_difficulty: bool = typer.Option(
        False, help="Optional filter difficulty, used for NUMINA."
    ),
    source: str = typer.Option(
        None, help="Source column filter for the dataset, used for NUMINA."
    ),
    result_dir: str = typer.Option("./", help="Result dir to save files."),
    check: bool = typer.Option(
        False, help="Perform evaluation checks on generated samples."
    ),
    inference: bool = typer.Option(False, help="Perform inference."),
    temperatures: List[float] = TEMP_DEFAULT,
    math_difficulty_lower_bound: int = typer.Option(
        None, help="Lowest difficulty level for math."
    ),
    math_difficulty_upper_bound: int = typer.Option(
        None, help="Highest difficulty level for math."
    ),
    system_prompt_template: str = typer.Option(
        None, help="System prompt template to use"
    ),
    n: int = typer.Option(1, help="Number of samples generated per problem."),
    seed: int = typer.Option(41, help="Random seed."),
    use_ray: bool = typer.Option(False, help="Use ray for scaling inference."),
    ray_config: str = typer.Option(
        None, help="Ray configuration file if using ray for scaling inference."
    ),
    ray_config_tensor_parallel_size: int = typer.Option(
        None,
        help="Ray configuration override for tensor parallel size per model replica",
    ),
    ray_config_num_replicas: int = typer.Option(
        None, help="Ray configuration override for number of model replicas"
    ),
    dtype: str = typer.Option(
        "float32", help="dtype for inference with vLLM.", case_sensitive=False
    ),
    top_p: float = typer.Option(1, help="Sampling parameter `top_p`"),
):
    # load ray config
    if use_ray:
        warnings.warn(
            "`tp` CLI argument is not compatible with `use-ray` and will be ignored. Please configure tensor parallel size in the `ray_config` YAML"
            " or override the value with the argument `ray-config-tensor-parallel-size` ",
            stacklevel=1,
        )
        if not ray_config:
            # load default
            ray_config = os.path.join(
                os.path.dirname(__file__), "ray_configs/ray_config.yaml"
            )
    set_seed(seed)

    # enable hf_transfer if not overridden by the user
    if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", None) is None:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    if task not in TASK_NAMES_TO_YAML:
        raise ValueError(
            f"Task {task} not found. Should be one of {TASK_NAMES_TO_YAML.keys()}"
        )

    task_config = TaskConfig.from_yaml(TASK_NAMES_TO_YAML[task])
    handler_name = task_config.handler
    handler_cls = TASK_HANDLER_MAP[handler_name]
    handler = handler_cls(task_config)

    model_config = ModelConfig.from_model_id(model, system_prompt_template)

    temperatures = [1] if model.startswith("openai/o1") else temperatures

    if top_p < 1 and model.startswith("openai/o1"):
        print(
            "OpenAI o1 models do not support `top_p` sampling. Resetting `top_p` to 1"
        )
        top_p = 1

    print(f"Temperature: {temperatures}")
    if temperatures == [0] and n > 1:
        n = 1
        print("Warning: Temperature 0 does not support multiple samples. Setting n=1.")

    # TODO: this can be cleaned up by allowing user override for any task_config with optional task_args
    # Currently kept here for consistency with old code
    split = split if split else handler.task_config.dataset_split
    subset = subset if subset else handler.task_config.dataset_subset
    if not difficulty and "difficulty" in handler.task_config.preprocess_config:
        difficulty = handler.task_config.preprocess_config["difficulty"]

    # create result dir if not exists
    if result_dir and not os.path.exists(result_dir):
        os.makedirs(result_dir)
    temperature_str = ",".join(map(str, temperatures))
    file_suffix = (
        f"{model_config.name}_{task}_{split}_subset_{subset}_filter_{filter_difficulty}"
        + f"_s{start}_e{end}_t{temperature_str}_n{n}"
    )
    if (
        math_difficulty_lower_bound is not None
        or math_difficulty_upper_bound is not None
    ):
        result_file = os.path.join(
            result_dir,
            f"{model_config.name}_{file_suffix}_{math_difficulty_upper_bound}.json",
        )
    else:
        result_file = os.path.join(
            result_dir,
            f"{file_suffix}.json",
        )
    # FIXME
    args = None
    if check:
        # check if converted file exists
        if (
            math_difficulty_lower_bound is not None
            or math_difficulty_upper_bound is not None
        ):
            converted_file = f"{result_dir}/converted_{file_suffix}.json"
        else:
            converted_file = f"{result_dir}/converted_{file_suffix}.json"
        if os.path.exists(converted_file):
            result_file = converted_file
        perform_check(handler, temperatures, result_file, args)
        return
    else:
        if use_ray:
            llm = None
        else:
            llm = (
                OpenAI()
                if model.startswith("openai")
                else LLM(model=model, tensor_parallel_size=tp, dtype=dtype)
            )
        if inference:
            perform_inference_and_save(
                handler, temperatures, max_tokens, result_file, llm, model_config, args
            )
        else:
            perform_inference_and_check(
                handler, temperatures, max_tokens, result_file, llm, model_config, args
            )


if __name__ == "__main__":
    app()
