"""
This is the recipe for data curation for the Sky T1 Preview model . 
"""

import argparse
import os

import datasets
import ray
from ray.data.llm import (
    HttpRequestProcessorConfig,
    build_llm_processor,
    vLLMEngineProcessorConfig,
)

from skythought.evals.scoring.math import MathEqualScorer

from .postprocess import APPSScorer, TACOScorer, convert_to_sharegpt_format
from .preprocess import APPSPreprocessor, NUMINAPreprocessor, TACOPreprocessor
from .prompts import CONVERT_PROMPT

parser = argparse.ArgumentParser()
parser.add_argument("--as-test", action="store_true")
args = parser.parse_args()

SYSTEM_PROMPT = "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step."  # noqa: E501

# 1. Load datasets
apps_ds = datasets.load_dataset("codeparrot/apps", split="test", streaming=True)
taco_ds_medium = datasets.load_dataset(
    "BAAI/TACO", split="test", name="MEDIUM", streaming=True
)
numina_ds = datasets.load_dataset("AI-MO/NuminaMath-CoT", split="train", streaming=True)

# convert all to ray dataset
apps_ds = ray.data.from_huggingface(apps_ds)
taco_ds_medium = ray.data.from_huggingface(taco_ds_medium)
numina_ds = ray.data.from_huggingface(numina_ds)


# get subsets from numina based on the source column
numina_ds_amc_aime = numina_ds.filter(lambda x: x["source"] == "amc_aime")
numina_ds_olympiads = numina_ds.filter(lambda x: x["source"] == "olympiads")
numina_ds_math = numina_ds.filter(lambda x: x["source"] == "math")


if args.as_test:
    apps_ds = apps_ds.limit(100)
    taco_ds_medium = taco_ds_medium.limit(100)
    numina_ds_amc_aime = numina_ds_amc_aime.limit(100)
    numina_ds_olympiads = numina_ds_olympiads.limit(100)
    numina_ds_math = numina_ds_math.limit(100)

# 2. Get model responses for each of the datasets
datasets = [
    apps_ds,
    taco_ds_medium,
    numina_ds_amc_aime,
    numina_ds_olympiads,
    numina_ds_math,
]

# these are user-defined simple preprocessing functions to go from entry -> prompt
preprocess_fns = [
    APPSPreprocessor(),
    TACOPreprocessor(),
    NUMINAPreprocessor(),
    NUMINAPreprocessor(),
    NUMINAPreprocessor(),
]

for i, ds in enumerate(datasets):
    datasets[i] = ds.map(preprocess_fns[i])

    # our API
    config = vLLMEngineProcessorConfig(
        # model="Qwen/QwQ-32B-Preview",
        model="Qwen/Qwen2-0.5B-Instruct",
        engine_kwargs=dict(
            enable_prefix_caching=True,
            enable_chunked_prefill=True,
            max_num_batched_tokens=16384,
        ),
        concurrency=2,
        batch_size=64,
    )

    # our API
    processor = build_llm_processor(
        config,
        preprocess=lambda row: dict(
            messages=[
                SYSTEM_PROMPT,
                {"role": "user", "content": row["user_input"]},
            ],
            sampling_params=dict(
                temperature=0.3,
                max_tokens=20,
                detokenize=False,
            ),
        ),
        postprocess=lambda row: dict(
            assistant_response=row["generated_text"],
            **row,  # This will return all the original columns in the dataset.
        ),
    )
    # our API
    datasets[i] = processor(ds)

# 3. Reformat the examples into a structured format
# define a configuration for the reformatter
config = HttpRequestProcessorConfig(
    url="https://api.openai.com/v1/chat/completions",
    headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
    # number of processors to run in parallel
    # Each handles a batch of requests
    concurrency=1,
)
# define the reformatter
reformatter = build_llm_processor(
    config=config,
    preprocess=lambda row: dict(
        # define the payload / the exact arguments to the OpenAI chat completions API
        payload=dict(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a solution format convertor."},
                {
                    "role": "user",
                    "content": CONVERT_PROMPT.format(
                        content=f"{row['question']}\n{row['assistant_response']}"
                    ),
                },
            ],
            temperature=0.7,
            max_tokens=16384,
        ),
    ),
    postprocess=lambda row: dict(
        formatted_response=row["http_response"]["choices"][0]["message"]["content"],
    ),
    batch_size=64,
)

for i, dataset in enumerate(datasets):
    datasets[i] = reformatter(dataset)


# 4. Rejection Sampling based on scoring
# apps, taco, numina-amc-aime, numina-olympiads, numina-math
numina_scorer = MathEqualScorer(
    response_key="formatted_response", answer_key="solution"
)
scorers = [APPSScorer(), TACOScorer(), numina_scorer, numina_scorer, numina_scorer]

for i, dataset in enumerate(datasets):
    fn = scorers[i]
    datasets[i] = dataset.map(fn)

# 5. Convert to ShareGPT format
for i, dataset in enumerate(datasets):
    datasets[i] = dataset.map(convert_to_sharegpt_format)

# 6. Union + Save datasets
datasets = datasets[0].union(*datasets[1:])
datasets.write_parquet("sky-t1-preview.parquet")
