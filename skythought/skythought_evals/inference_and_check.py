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
                add_generation_prompt=model_config.assistant_prefill is None,
                continue_final_message=model_config.assistant_prefill is not None,
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
    output_dir: Path,
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
                prompt = conversations[idx][1]["content"]
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
    summary_file = output_dir / "summary.json"

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
    result_file = output_dir / "results.json"
    with open(result_file, "w", encoding="utf-8") as file:
        json.dump(id_to_results, file, ensure_ascii=False, indent=4, cls=NumpyEncoder)


def score_results(handler: TaskHandler, run_dir: Path, run_summary: dict):
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
            prompt = conversations[idx][1]["content"]
            results[unique_id]["prompt"] = prompt

        results[unique_id]["responses"] = response_entries

        results[unique_id]["token_usages"] = token_usages

    # Prepare the summary dictionary
    summary_dict = {
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

    # Save the summary dictionary to the result file
    with open(summary_file, "w") as f:
        json.dump(summary_dict, f, indent=4)

    print(f"Summary saved to {summary_file}")

    with open(result_file, "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=4, cls=NumpyEncoder)

    print(f"Results saved to {result_file}")
