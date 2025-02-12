import concurrent.futures
import copy
import json
import logging
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import ray
from openai import OpenAI
from skythought_evals.batch import Pipeline, init_engine_from_config
from skythought_evals.batch.env_config import EnvConfig
from skythought_evals.batch.workload import EvalWorkload
from skythought_evals.common.entities import (
    Backend,
    BackendParameters,
    RayLLMEngineArgs,
    SamplingParameters,
)
from skythought_evals.models import ModelConfig
from skythought_evals.tasks import (
    NUMINATaskHandler,
    TaskHandler,
)
from skythought_evals.util.metrics import pass_at_k
from skythought_evals.util.response import Response, SingleParsedResponse
from tqdm import tqdm
from vllm import LLM

logger = logging.getLogger(__name__)
module_dir = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RAY_CONFIG_RELATIVE_PATH = "ray_configs/ray_config.yaml"


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# temperature: float = TEMPERATURE_DEFAULT
#     top_p: float = TOP_P_DEFAULT
#     n: int = 1
#     max_tokens: int = MAX_TOKENS_DEFAULT
#     reasoning_effort: Optional[float] = None
#     frequency_penalty: Optional[float] = None


def fetch_response_openai(
    client, model_config, sampling_params: SamplingParameters, prompt
):
    model_name = model_config.model_id.replace("openai/", "")
    if "o1" in model_name:
        # O1 doesn't support system prompt
        # NOTE: might want to implement this inside handler instead
        for p in prompt:
            p["role"] = "user"

        response = client.chat.completions.create(
            model=model_name,
            messages=prompt,
            n=sampling_params.params.n,
            temperature=sampling_params.params.temperature,
            max_tokens=sampling_params.params.max_tokens,
            reasoning_effort=sampling_params.params.reasoning_effort,
            frequency_penalty=sampling_params.params.frequency_penalty,
            max_completion_tokens=sampling_params.params.max_tokens,
        )
    else:
        response = client.chat.completions.create(
            model=model_name,
            messages=prompt,
            n=sampling_params.params.n,
            temperature=sampling_params.params.temperature,
            max_tokens=sampling_params.params.max_tokens,
            reasoning_effort=sampling_params.params.reasoning_effort,
            frequency_penalty=sampling_params.params.frequency_penalty,
            max_completion_tokens=sampling_params.params.max_tokens,
        )
    return response


def fetch_responses_ray(
    conversations,
    backend_args: RayLLMEngineArgs,
    model_config: ModelConfig,
    sampling_params: SamplingParameters,
):
    config = backend_args.get_ray_llm_config()
    config["model_id"] = model_config.model_id
    # use user-provided dtype from CLI
    config["engine_kwargs"]["dtype"] = model_config.dtype

    engine_cfg = init_engine_from_config(config)
    ds = ray.data.from_items([(idx, conv) for idx, conv in enumerate(conversations)])
    num_replicas = config["env_config"].get("num_replicas", 1)
    if ds.count() < config["env_config"].get("batch_size", 1):
        config["env_config"]["batch_size"] = math.ceil(ds.count() / num_replicas)
    if num_replicas > 1 and num_replicas > ds.num_blocks():
        ds = ds.repartition(num_partitions=num_replicas)
    # {
    #     "n": args.n,
    #     "max_tokens": max_tokens,
    #     "temperature": temp,
    #     "top_p": args.top_p,
    # }
    workload = EvalWorkload(
        dataset=ds,
        sampling_params=sampling_params.to_dict(),
    )
    pipeline = Pipeline(
        engine_cfg,
        env_config=EnvConfig(**config["env_config"]),
    )
    ds = pipeline(workload)
    responses = ds.materialize()
    return responses


def _parse_response_for_idx(
    response: Response,
    sample_idx: int,
) -> Tuple[SingleParsedResponse, Dict[str, int]]:
    content = response.response[sample_idx].strip()
    response_entry = SingleParsedResponse(content=content)

    token_usage_for_response = {
        "completion_tokens": response.num_completion_tokens[sample_idx],
        "prompt_tokens": response.num_input_tokens,
    }
    return response_entry, token_usage_for_response


def inference(
    conversations,
    backend,
    backend_params: BackendParameters,
    model_config,
    sampling_params: SamplingParameters,
    **kwargs,
):
    if backend == "ray":
        responses = fetch_responses_ray(
            conversations, backend_params, model_config, sampling_params
        )
        responses = [
            Response.from_ray_response(response) for response in responses.iter_rows()
        ]
        # NOTE: This deepcopy is needed to avoid a SIGSEV error related to object cleanup with the ray object store and
        # the later use of ProcessPoolExecutor - see here: https://github.com/NovaSky-AI/SkyThought/pull/63#discussion_r1941899714
        # TODO: revisit the underlying issue and remove the deepcopy if possible
        responses = copy.deepcopy(responses)
        responses = sorted(responses, key=lambda x: x.index)
    elif backend == "openai":
        llm = OpenAI(**backend_params)
        fetch_partial = partial(
            fetch_response_openai,
            llm,
            model_config,
            sampling_params,
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as e:
            responses = list(e.map(fetch_partial, conversations))

        responses = [Response.from_openai_response(response) for response in responses]
    elif backend == "vllm":
        batch_size = kwargs.get("batch_size", 1)
        engine_kwargs = copy.deepcopy(backend_params.to_dict())
        engine_kwargs["model"] = model_config.model_id
        llm = LLM(**engine_kwargs)

        response_in_batches = [
            llm.chat(
                messages=conversations[i : i + batch_size],
                sampling_params=sampling_params.params,
                use_tqdm=True,
            )
            for i in range(0, len(conversations), batch_size)
        ]
        responses = []
        for response_batch in response_in_batches:
            responses.extend(response_batch)
        responses = [Response.from_vllm_response(response) for response in responses]
    else:
        raise ValueError(f"Invalid backend: {backend}")
    # flush logs to stdout
    sys.stdout.flush()

    return responses


def load_existing_results(result_file):
    if not os.path.exists(result_file):
        return {}
    with open(result_file, "r", encoding="utf-8") as f:
        records = json.load(f)
    return records


def generate_and_score(
    handler: TaskHandler,
    model_config: ModelConfig,
    backend: Backend,
    backend_params: BackendParameters,
    sampling_params: SamplingParameters,
    output_dir: str,
    start: int,
    end: int,
    run_config_dict: dict,
    **kwargs,
):
    eval_data = handler.load_and_filter_dataset(start, end).to_dict(orient="records")

    conversations = handler.make_conversations(
        eval_data, model_config.system_prompt, model_config.user_template
    )
    unique_ids = [eval_data[i]["_index"] for i in range(len(eval_data))]

    if len(conversations) == 0:
        print("No more data to process")
        return

    responses = inference(
        conversations, backend, backend_params, model_config, sampling_params, **kwargs
    )

    total_correct = 0
    total_finish = 0
    # temperature_to_scores[temp] = {}
    # we will have the following objects
    id_to_scores: Dict[int, List[bool]] = {}
    # FIXME: we should start tracking final outputs for each problem. Downstream use-case is for metrics like cons@k
    # id_to_final_outputs: Dict[int, List[str]] = {}
    id_to_results: Dict[int, Dict[str, Any]] = {}

    with ProcessPoolExecutor(max_workers=32) as executor:
        future_to_task = {}
        token_usages = {}
        for idx, response in enumerate(responses):
            for sample_idx in range(sampling_params.params.n):
                # response_entry at this point doesn't contain correctness check.
                response_entry, token_usage_for_response = _parse_response_for_idx(
                    response,
                    sample_idx,
                )
                unique_id = unique_ids[idx]
                if unique_id not in token_usages:
                    token_usages[unique_id] = []
                token_usages[unique_id].append(token_usage_for_response)
                # submit correctness check for response
                future_to_task[
                    executor.submit(
                        handler.update_results,
                        eval_data[idx],
                        response_entry.content,
                    )
                ] = (idx, sample_idx)

        for future in tqdm(
            as_completed(future_to_task),
            total=len(future_to_task),
            desc="Processing Generations",
        ):
            idx, sample_idx = future_to_task[future]
            # TODO (sumanthrh): the returned entry is currently a dict and can be confusing.
            # this should also be a ParsedResponse object.
            response_entry: dict = future.result()
            total_correct += response_entry["correctness"]
            total_finish += 1

            unique_id = unique_ids[idx]
            if unique_id not in id_to_results:
                id_to_results[unique_id] = eval_data[idx]
                if isinstance(handler, NUMINATaskHandler):
                    id_to_results[unique_id]["messages"] = ""
                id_to_results[unique_id]["responses"] = []
                id_to_results[unique_id]["token_usages"] = []
                prompt = conversations[idx][-1]["content"]
                id_to_results[unique_id]["prompt"] = prompt
                id_to_results[unique_id]["input_conversation"] = conversations[idx]
                id_to_scores[unique_id] = [0 for _ in range(sampling_params.params.n)]

            if not len(id_to_results[unique_id]["responses"]):
                id_to_results[unique_id]["responses"] = [
                    {} for _ in range(sampling_params.params.n)
                ]

            id_to_results[unique_id]["responses"][sample_idx] = response_entry
            # do this only once per problem/idx
            if not len(id_to_results[unique_id]["token_usages"]):
                id_to_results[unique_id]["token_usages"] = token_usages[unique_id]

            # update scores
            id_to_scores[unique_id][sample_idx] = response_entry["correctness"]

        print(f"Final acc: {total_correct}/{total_finish}")

        acc = round(total_correct / total_finish, 4) if total_finish > 0 else 0
        print(json.dumps({"acc": acc}))

    pass_at_k_metrics = None
    if sampling_params.params.n > 1:
        pass_at_k_metrics = pass_at_k(sampling_params.params.n, id_to_scores)
        print(json.dumps({"pass_at_k": pass_at_k_metrics}))

    total_prompt_tokens = sum(
        id_to_results[key]["token_usages"][sample_idx]["prompt_tokens"]
        for sample_idx in range(sampling_params.params.n)
        for key in id_to_results
    )
    total_completion_tokens = sum(
        id_to_results[key]["token_usages"][sample_idx]["completion_tokens"]
        for sample_idx in range(sampling_params.params.n)
        for key in id_to_results
    )
    num_responses_total = len(responses) * sampling_params.params.n

    # Run summary
    summary_file = os.path.join(output_dir, "summary.json")

    # Prepare the summary dictionary
    summary_dict = {
        "configuration": run_config_dict,
        "completion_tokens": total_completion_tokens,
        "prompt_tokens": total_prompt_tokens,
        "avg_completion_tokens": (
            round(total_completion_tokens / num_responses_total, 3)
            if total_completion_tokens
            else 0
        ),
        "avg_prompt_tokens": (
            round(total_prompt_tokens / num_responses_total, 3)
            if total_prompt_tokens
            else 0
        ),
        "pass_at_k": pass_at_k_metrics,
        "accuracy": acc,
    }

    # Save the token usage dictionary to the result file
    with open(summary_file, "w") as f:
        json.dump(summary_dict, f, indent=4)

    print(f"Summary saved to {summary_file}")
    result_file = os.path.join(output_dir, "results.json")
    with open(result_file, "w", encoding="utf-8") as file:
        json.dump(id_to_results, file, ensure_ascii=False, indent=4, cls=NumpyEncoder)


def score_results(handler: TaskHandler, run_dir: Path, task: str, run_summary: dict):
    result_file = run_dir / "results.json"
    results = load_existing_results(result_file)
    print(f"Loaded {len(results)} existing results.")

    total_correct = 0
    total_finish = 0
    id_to_scores = {}
    num_generations_per_problem = len(results[list(results.keys())[0]]["responses"])

    with ProcessPoolExecutor(max_workers=32) as executor:
        future_to_task = {
            executor.submit(
                handler.update_results, result, result["responses"][i]["content"]
            ): (
                result,
                i,
                unique_id,
            )
            for unique_id, result in results.items()
            for i in range(num_generations_per_problem)
        }

        # Collect results as they finish
        for future in tqdm(
            as_completed(future_to_task),
            total=len(future_to_task),
            desc="Scoring results",
        ):
            result, sample_idx, unique_id = future_to_task[future]
            new_response_entry = future.result()
            total_correct += new_response_entry["correctness"]
            total_finish += 1

            # Update the corresponding record in results
            if unique_id not in id_to_scores:
                id_to_scores[unique_id] = [0 for _ in range(len(result["responses"]))]

            results[unique_id]["responses"][sample_idx]["correctness"] = (
                new_response_entry["correctness"]
            )
            results[unique_id]["responses"][sample_idx]["reason"] = new_response_entry[
                "reason"
            ]
            id_to_scores[unique_id][sample_idx] = new_response_entry["correctness"]

    acc = round(total_correct / total_finish, 4) if total_finish > 0 else 0
    print(f"Final reject-sampling accuracy: {acc}")

    pass_at_k_metrics = None
    if len(results) > 1:
        pass_at_k_metrics = pass_at_k(num_generations_per_problem, id_to_scores)
        print(json.dumps({"pass_at_k": pass_at_k_metrics}))

    run_summary.update({"accuracy": acc, "pass_at_k": pass_at_k_metrics})
    # save summary
    summary_file = run_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(run_summary, f, indent=4)

    with open(result_file, "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=4, cls=NumpyEncoder)

    logging.info(f"Saved results to {str(result_file)}")


def perform_check(handler: TaskHandler, temperatures, result_file, args):
    results = load_existing_results(result_file)
    print(f"Loaded {len(results)} existing results.")

    train_data = handler.load_and_filter_dataset(
        args.start,
        args.end,
        split=args.split,
        subset=args.subset,
        difficulty=args.difficulty,
        args=args,
    )
    remaining_data = handler.process_remaining_data(train_data, {})

    tasks = []
    for item in remaining_data:
        problem_key = item[handler.question_key]
        # If this item exists in the results file, check each temperature
        if problem_key in results and "responses" in results[problem_key]:
            for temp in temperatures:
                if str(temp) in results[problem_key]["responses"]:
                    response_entries = results[problem_key]["responses"][str(temp)]
                    for sample_id, response_entry in enumerate(response_entries):
                        if sample_id > (args.n - 1):
                            continue
                        if True or response_entry["correctness"] is None:
                            processed = "processed_content" in response_entry
                            tasks.append(
                                (
                                    item,
                                    temp,
                                    (
                                        response_entry["processed_content"]
                                        if processed
                                        else response_entry["content"]
                                    ),
                                    sample_id,
                                )
                            )

    print(f"Found {len(tasks)} responses requiring reject sampling...")

    total_correct = 0
    total_finish = 0
    correct = {temp: {} for temp in temperatures}
    with ProcessPoolExecutor(max_workers=32) as executor:
        future_to_task = {
            executor.submit(handler.update_results, item, content): (
                item,
                temp,
                sample_id,
            )
            for (item, temp, content, sample_id) in tasks
        }

        # 4. Collect the results as they finish.
        for future in tqdm(
            as_completed(future_to_task),
            total=len(future_to_task),
            desc="Processing Reject Sampling",
        ):
            item, temp, sample_id = future_to_task[future]
            new_response_entry = future.result()
            total_correct += new_response_entry["correctness"]
            total_finish += 1

            # Update the corresponding record in results
            problem_key = item[handler.question_key]
            if problem_key not in correct[temp]:
                correct[temp][problem_key] = False
            if new_response_entry["correctness"]:
                correct[temp][problem_key] = True
            assert (
                problem_key in results
                and "responses" in results[problem_key]
                and str(temp) in results[problem_key]["responses"]
            )
            response_entry = results[problem_key]["responses"][str(temp)][sample_id]
            response_entry["correctness"] = new_response_entry["correctness"]
            response_entry["reason"] = new_response_entry["reason"]
            results[problem_key]["responses"][str(temp)][sample_id] = response_entry

    print(f"Final reject-sampling accuracy: {total_correct}/{total_finish}")
    # per temperature acc
    for temp in temperatures:
        temp_correct = sum(correct[temp].values())
        temp_total = len(correct[temp])
        temp_acc = round(temp_correct / temp_total, 4) if temp_total > 0 else 0
        print(f"Temperature {temp} acc: {temp_correct}/{temp_total} ({temp_acc})")

    with open(result_file, "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=4, cls=NumpyEncoder)


def generate_and_save(
    handler: TaskHandler,
    model_config: ModelConfig,
    backend: Backend,
    backend_params: BackendParameters,
    sampling_params: SamplingParameters,
    output_dir: Path,
    start: int,
    end: int,
    run_config_dict: dict,
    resume_from: os.PathLike | None = None,
    **kwargs,
):
    results = {}
    if resume_from is not None:
        resume_from = Path(resume_from)
        result_file = resume_from / "results.json"
        summary_file = resume_from / "summary.json"
        results = load_existing_results(result_file)
        print(f"Loaded {len(results)} existing results.")
    else:
        result_file = output_dir / "results.json"
        summary_file = output_dir / "summary.json"

    full_eval_data = handler.load_and_filter_dataset(start, end)
    remaining_data = handler.process_remaining_data(full_eval_data, results)
    print(f"Generating results for {len(remaining_data)} problems")

    unique_ids = [remaining_data[i]["_index"] for i in range(len(remaining_data))]

    if not len(remaining_data):
        print("All results saved. Exiting...")
        return
    conversations = handler.make_conversations(
        remaining_data, model_config.system_prompt, model_config.user_template
    )

    if len(conversations) == 0:
        print("No more data to process")
        return

    responses = inference(
        conversations, backend, backend_params, model_config, sampling_params, **kwargs
    )

    completion_tokens = []
    prompt_tokens = []
    for idx, response in enumerate(responses):
        response_entries = []
        token_usages = []
        completion_token = 0
        for sample_idx in range(sampling_params.params.n):
            response_entry, token_usage_for_response = _parse_response_for_idx(
                response,
                sample_idx,
            )
            token_usages.append(token_usage_for_response)
            completion_token += token_usage_for_response["completion_tokens"]
            response_entries.append(response_entry.to_dict())

        completion_token /= sampling_params.params.n
        prompt_token = response.num_input_tokens
        prompt_tokens.append(prompt_token)
        completion_tokens.append(completion_token)

        unique_id = unique_ids[idx]
        if unique_id not in results:
            results[unique_id] = remaining_data[idx]
            if isinstance(handler, NUMINATaskHandler):
                results[unique_id]["messages"] = ""
            prompt = conversations[idx][-1]["content"]
            results[unique_id]["prompt"] = prompt

        results[unique_id]["responses"] = response_entries

        results[unique_id]["token_usages"] = token_usages

    # Prepare the token usage dictionary
    metrics_dict = {
        "configuration": run_config_dict,
        "completion_tokens": sum(completion_tokens),
        "prompt_tokens": sum(prompt_tokens),
        "avg_completion_tokens": (
            round(sum(completion_tokens) / len(completion_tokens), 3)
            if completion_tokens
            else 0
        ),
        "avg_prompt_tokens": (
            round(sum(prompt_tokens) / len(prompt_tokens), 3) if prompt_tokens else 0
        ),
    }

    # Save the usage dictionary to the result file
    with open(summary_file, "w") as f:
        json.dump(metrics_dict, f, indent=4)

    print(f"Summary saved to {summary_file}")

    with open(result_file, "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=4, cls=NumpyEncoder)

    print(f"Results saved to {result_file}")


# def main():
#     parser = argparse.ArgumentParser(
#         description="Unified inference and checking for different datasets/tasks."
#     )
#     parser.add_argument(
#         "--task",
#         type=str,
#         required=True,
#         choices=TASK_NAMES_TO_YAML.keys(),
#         help="Task to process.",
#     )
#     parser.add_argument(
#         "--model",
#         type=str,
#         required=True,
#         default="Qwen/QwQ-32B-Preview",
#         help="The model to run.",
#     )
#     parser.add_argument("--tp", type=int, default=8, help="Tensor Parallelism Degree")
#     parser.add_argument(
#         "--max_tokens", type=int, default=32768, help="Max tokens for the model."
#     )
#     parser.add_argument(
#         "--split",
#         type=str,
#         default=None,
#         help="Split to use for the dataset (e.g., train, test).",
#     )
#     parser.add_argument("--subset", type=str, help="Subset for the dataset.")
#     parser.add_argument("--start", type=int, default=0, help="Start index.")
#     parser.add_argument("--end", type=int, default=-1, help="End index.")
#     parser.add_argument(
#         "--difficulty",
#         type=str,
#         default=None,
#         help="Difficulty level. Example: 'easy', 'medium', 'hard'.",
#     )
#     parser.add_argument(
#         "--filter-difficulty",
#         action="store_true",
#         help="Optional filter difficulty, used for NUMINA.",
#     )
#     parser.add_argument(
#         "--source",
#         type=str,
#         help="Source column filter for the dataset, used for NUMINA.",
#     )
#     parser.add_argument(
#         "--result-dir", type=str, default="./", help="Result dir to save files."
#     )
#     parser.add_argument(
#         "--check",
#         action="store_true",
#         help="Perform evaluation checks on generated samples.",
#     )
#     parser.add_argument("--inference", action="store_true", help="Perform inference.")
#     parser.add_argument(
#         "--temperatures",
#         type=float,
#         nargs="+",
#         default=[0],
#         help="Temperature for sampling.",
#     )
#     parser.add_argument(
#         "--math-difficulty-lower-bound",
#         type=int,
#         default=None,
#         help="Lowest difficulty level for math.",
#     )
#     parser.add_argument(
#         "--math-difficulty-upper-bound",
#         type=int,
#         default=None,
#         help="Highest difficulty level for math.",
#     )
#     parser.add_argument(
#         "--system-prompt-template",
#         type=str,
#         default=None,
#         help="System prompt template to use",
#         choices=get_system_prompt_keys(),
#     )
#     parser.add_argument(
#         "--n", type=int, default=1, help="Number of samples generated per problem."
#     )
#     parser.add_argument("--seed", type=int, default=41, help="Random seed.")
#     parser.add_argument(
#         "--use-ray", action="store_true", help="Use ray for scaling inference."
#     )
#     parser.add_argument(
#         "--ray-config",
#         type=str,
#         default=None,
#         help="Ray configuration file if using ray for scaling inference. By default, we use the example in ray_configs/ray_config.yaml",
#     )
#     parser.add_argument(
#         "--ray-config-tensor-parallel-size",
#         type=int,
#         default=None,
#         help="Ray configuration override for tensor parallel size per model replica",
#     )
#     parser.add_argument(
#         "--ray-config-num-replicas",
#         type=int,
#         default=None,
#         help="Ray configuration override for number of model replicas",
#     )
#     parser.add_argument(
#         "--dtype",
#         type=str,
#         choices=["float32", "auto", "float16", "bfloat16"],
#         help="dtype for inference with vLLM. Full-precision by default."
#         "'auto' refers to automatically inferring dtype for the model",
#         default="float32",
#     )
#     parser.add_argument(
#         "--top_p",
#         type=float,
#         default=1,
#         help="Sampling parameter `top_p`",
#     )
#     args = parser.parse_args()
#     # load ray config
#     if args.use_ray:
#         warnings.warn(
#             "`tp` CLI argument is not compatible with `use-ray` and will be ignored. Please configure tensor parallel size in the `ray_config` YAML"
#             " or override the value with the argument `ray-config-tensor-parallel-size` ",
#             stacklevel=1,
#         )
#         if not args.ray_config:
#             # load default
#             args.ray_config = os.path.join(module_dir, DEFAULT_RAY_CONFIG_RELATIVE_PATH)
#     set_seed(args.seed)

#     # enable hf_transfer if not overriden by the user
#     if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", None) is None:
#         os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

#     if args.task not in TASK_NAMES_TO_YAML:
#         raise ValueError(
#             f"Task {args.task} not found. Should be one of {TASK_NAMES_TO_YAML.keys()}"
#         )

#     task_config = TaskConfig.from_yaml(TASK_NAMES_TO_YAML[args.task])
#     handler_name = task_config.handler
#     handler_cls = TASK_HANDLER_MAP[handler_name]
#     handler = handler_cls(task_config)

#     model_config = ModelConfig.from_model_id(args.model, args.system_prompt_template)

#     temperatures = [1] if args.model.startswith("openai/o1") else args.temperatures

#     if args.top_p < 1 and args.model.startswith("openai/o1"):
#         print(
#             "OpenAI o1 models do not support `top_p` sampling. Resetting `top_p` to 1"
#         )
#         args.top_p = 1

#     print(f"Temperature: {temperatures}")
#     max_tokens = args.max_tokens
#     if temperatures == [0] and args.n > 1:
#         args.n = 1
#         print("Warning: Temperature 0 does not support multiple samples. Setting n=1.")

#     # TODO: this can be cleaned up by allowing user override for any task_config with optional task_args
#     # Currently kept here for consistency with old code
#     args.split = args.split if args.split else handler.task_config.dataset_split
#     args.subset = args.subset if args.subset else handler.task_config.dataset_subset
#     if not args.difficulty and "difficulty" in handler.task_config.preprocess_config:
#         args.difficulty = handler.task_config.preprocess_config["difficulty"]

#     # create result dir if not exists
#     if args.result_dir and not os.path.exists(args.result_dir):
#         os.makedirs(args.result_dir)
#     temperature_str = ",".join(map(str, temperatures))
#     file_suffix = (
#         f"{model_config.name}_{args.task}_{args.split}_subset_{args.subset}_filter_{args.filter_difficulty}"
#         + f"_s{args.start}_e{args.end}_t{temperature_str}_n{args.n}"
#     )
#     if (
#         args.math_difficulty_lower_bound is not None
#         or args.math_difficulty_upper_bound is not None
#     ):
#         result_file = os.path.join(
#             args.result_dir,
#             f"{model_config.name}_{file_suffix}_{args.math_difficulty_upper_bound}.json",
#         )
#     else:
#         result_file = os.path.join(
#             args.result_dir,
#             f"{file_suffix}.json",
#         )

#     if args.check:
#         # check if converted file exists
#         if (
#             args.math_difficulty_lower_bound is not None
#             or args.math_difficulty_upper_bound is not None
#         ):
#             converted_file = f"{args.result_dir}/converted_{file_suffix}.json"
#         else:
#             converted_file = f"{args.result_dir}/converted_{file_suffix}.json"
#         if os.path.exists(converted_file):
#             result_file = converted_file
#         perform_check(handler, temperatures, result_file, args)
#         return
#     else:
#         if args.use_ray:
#             llm = None
#         else:
#             llm = (
#                 OpenAI()
#                 if args.model.startswith("openai")
#                 else LLM(
#                     model=args.model, tensor_parallel_size=args.tp, dtype=args.dtype
#                 )
#             )
#         if args.inference:
#             perform_inference_and_save(
#                 handler, temperatures, max_tokens, result_file, llm, model_config, args
#             )
#         else:
#             perform_inference_and_check(
#                 handler, temperatures, max_tokens, result_file, llm, model_config, args
#             )


# if __name__ == "__main__":
#     main()
