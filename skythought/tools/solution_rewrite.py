import argparse
import json
import os
from tqdm import tqdm
from util.math.testing_util import strip_answer_string
from util.model_utils import *
from vllm import LLM, SamplingParams

def load_dataset(dataset_path : str):
  data = {}
  with open(dataset_path, 'r', encoding='utf-8') as file:
    data = json.load(file)
  return data


def make_scoring_conversations(dataset, system_prompt):
  conversations = []
  for _, key in enumerate(dataset):
    problem = dataset[key]
    gt_answer = strip_answer_string(problem["answer"])
    for response_key in problem["responses"]:
      response = problem["responses"][response_key]["content"]
      prompt_text = response + "\n#####\nThe ground truth answer is " + gt_answer
      conversations.append([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt_text}
      ])

  return conversations

def score_solutions(dataset, responses, outfile):
  idx = 0
  for _, key in tqdm(enumerate(dataset), total=len(dataset), desc="Scoring original solutions"):
    problem = dataset[key]
    for response_key in problem["responses"]:
      score = responses[idx].outputs[0].text.strip()
      problem["responses"][response_key]["correctness"] = (score == "True")
      idx += 1

  with open(outfile, 'w', encoding='utf-8') as new_file:
      json.dump(dataset, new_file, ensure_ascii=False, indent=2)
  return dataset


def filter_solutions(scored_dataset):
  # Filter correct solutions and ensure each problem has at least 2 correct responses
  filtered_dataset = {
    key: {
      "responses": {
        r_key: r_value
        for r_key, r_value in problem["responses"].items()
        if r_value["correctness"]
      },
      "token_usages": {
        t_key: t_value
        for t_key, t_value in problem["token_usages"].items()
        if t_key in problem["responses"] and problem["responses"][t_key]["correctness"]
      },
    }
    for key, problem in scored_dataset.items()
    if sum(r["correctness"] for r in problem["responses"].values()) >= 2
  }

  # Extract shortest and longest correct solutions
  for _, problem in filtered_dataset.items():
    token_usages = problem["token_usages"]
    shortest_key, shortest_entry = min(token_usages.items(), key=lambda x: x[1]["completion_tokens"])
    longest_key, longest_entry = max(token_usages.items(), key=lambda x: x[1]["completion_tokens"])
    problem["token_usages"] = {
      "shortest": shortest_entry,
      "longest": longest_entry,
    }
    problem["responses"] = {
      "shortest": problem["responses"][shortest_key],
      "longest": problem["responses"][longest_key],
    }

    return filtered_dataset


def make_splitting_conversations(data, system_prompt):
  conversations = []
  for problem in data:
    response = data[problem]["responses"]["shortest"]
    prompt_text = response["content"]
    conversations.append([
      {"role": "system", "content": system_prompt},
      {"role": "user", "content": prompt_text}
    ])
  return conversations


def split_solutions(dataset, responses, delimiter):
  outputs = []
  for _, response in tqdm(enumerate(responses), total=len(responses), desc="Splitting responses"):
    content = response.outputs[0].text.strip()
    # Split response by configured delimiter.
    split_content = content.split(delimiter)
    split_content = [x.strip() for x in split_content if x != ""]
    outputs.append(split_content)
  for idx, key in enumerate(dataset):
    solutions = outputs[idx]
    problem = dataset[key]
    problem["responses"]["shortest"]["subsolutions"] = solutions
  return dataset

def make_subscoring_conversations(dataset, system_prompt):
  conversations = []
  for _, key in enumerate(dataset):
    problem = dataset[key]
    gt_answer = strip_answer_string(problem["answer"])
    subsolutions = problem["responses"]["shortest"]["subsolutions"]
    for sub in subsolutions:
      prompt_text = sub + "\n#####\nThe ground truth answer is " + gt_answer
      conversations.append([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt_text}
      ])

  return conversations


def score_subsolutions(dataset, responses):
  idx = 0
  for _, key in tqdm(enumerate(dataset), total=len(dataset), desc="Scoring sub-solutions"):
    problem = dataset[key]
    subsolutions = problem["responses"]["shortest"]["subsolutions"]
    scores = []
    for _, sub in enumerate(subsolutions):
      score = responses[idx].outputs[0].text.strip()
      scores.append(score == "True")
      idx += 1
    problem["responses"]["shortest"]["scores"] = scores
  return dataset


def build_response_variants(dataset):
  def clean_response_string(response):
    if '<|end_of_thought|>' not in response:
      response += '<|end_of_thought|>'
    return response

  keys_to_remove = []

  for key, problem in dataset.items():
    scores = problem["responses"]["shortest"]["scores"]
    subsolutions = problem["responses"]["shortest"]["subsolutions"]

    # Check if there are valid scores
    if True not in scores:
      keys_to_remove.append(key)
      continue

    # Build FCS (First Correct Solution)
    fcs_idx = scores.index(True)
    fcs_response = "\n".join(subsolutions[:fcs_idx + 1]) if fcs_idx < len(scores) - 1 else "\n".join(subsolutions[:-1])
    fcs_response = clean_response_string(fcs_response) + "\n" + subsolutions[-1]
    problem["responses"]["fcs"] = fcs_response

    # Build FCS + 1
    fcs_plus1_idx = fcs_idx + 1 if fcs_idx + 1 < len(subsolutions) - 1 else fcs_idx
    fcs_plus1_response = "\n".join(subsolutions[:fcs_plus1_idx + 1])
    fcs_plus1_response = clean_response_string(fcs_plus1_response) + "\n" + subsolutions[-1]
    problem["responses"]["fcs_plus1"] = fcs_plus1_response

    # Check if there are valid scores
    if True not in scores[fcs_idx + 1:]:
      keys_to_remove.append(key)
      continue

    # Build FCS + Reflection
    fcs_reflection_idx = scores.index(True, fcs_idx + 1)
    fcs_reflection_response = "\n".join(subsolutions[:fcs_reflection_idx + 1]) if fcs_reflection_idx < len(scores) - 1 else "\n".join(subsolutions[:-1])
    fcs_reflection_response = clean_response_string(fcs_reflection_response) + "\n" + subsolutions[-1]
    problem["responses"]["fcs_reflection"] = fcs_reflection_response

  # Remove problems without valid sub-solutions
  for key in keys_to_remove:
    del dataset[key]

  return dataset


def compute_token_usages(dataset, llm):
  tokenizer = llm.get_tokenizer()
  for key in tqdm(dataset, desc="Computing token usages", total=len(dataset)):
    problem = dataset[key]
    prompt_tokens = problem["token_usages"]["shortest"]["prompt_tokens"]
    for variant in ["fcs", "fcs_plus1", "fcs_reflection"]:
      problem["token_usages"][variant] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": len(tokenizer(problem["responses"][variant]).input_ids)
      }
  return dataset

def build_question_prompt(prompt):
  return "Return your final response within \\boxed{{}}" + prompt

def make_preference_conversations(final_dataset, format, system_prompt):
  conversations = []
  for prompt in final_dataset:
    problem = final_dataset[prompt]
    convo = {}
    convo["conversations"] = [
      {
        "from": "system",
        "value": system_prompt,
      },
      {
        "from": "human",
        "value": build_question_prompt(prompt),
      }
    ]
    convo["chosen"] = {
      "from": "gpt",
      "value": problem["responses"][format],
    }
    convo["rejected"] = {
      "from": "gpt",
      "value": problem["responses"]["longest"]["content"]
    }
    conversations.append(convo)

  return conversations

def main():
  parser = argparse.ArgumentParser(description="Filter, rewrite, and format generated responses for high-quality data curation.")
  parser.add_argument("--rewrite-model", type=str, required=True, default="meta-llama/Llama-3.3-70B-Instruct", help="The model used for response processing.")
  parser.add_argument("--target-model", type=str, required=True, default="NovaSky-AI/Sky-T1-32B-Preview", help="The target model the rewritten responses will be used to train.")
  parser.add_argument("--dataset", type=str, required=True, help="Path to the starting dataset of generated responses to filter from.")
  parser.add_argument("--result-dir", type=str, default="./", help="Result directory to save processed data.")
  parser.add_argument("--checkpoint", action="store_true", help="Whether to checkpoint the dataset at each step.")
  parser.add_argument("--tp", type=int, default=8, help="Tensor Parallelism Degree")
  parser.add_argument("--max_tokens", type=int, default=32768, help="Max tokens for the model.")
  args = parser.parse_args()
  
  if args.result_dir and not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

  # Initialize model for data processing.
  llm = LLM(model=args.rewrite_model, tensor_parallel_size=args.tp)
  sampling_params = SamplingParams(max_tokens=args.max_tokens)
  
  dataset = load_dataset(args.dataset)
  
  # Filter for the shortest and longest correct solutions.
  filtered_dataset = filter_solutions(dataset)
  if args.checkpoint:
    outfile = os.path.join(args.result_dir, f"filtered-responses.json")
    with open(outfile, 'w', encoding='utf-8') as new_file:
        json.dump(filtered_dataset, new_file, ensure_ascii=False, indent=2)

  # Split the shortest solution into subsolutions using the configured model.
  conversations = make_splitting_conversations(filtered_dataset, SUBPROBLEM_SPLIT_PROMPT)
  responses = llm.chat(messages=conversations, sampling_params=sampling_params, use_tqdm=True)
  split_dataset = split_solutions(filtered_dataset, responses, '#####')
  if args.checkpoint:
    outfile = os.path.join(args.result_dir, f"split-solutions.json")
    with open(outfile, 'w', encoding='utf-8') as new_file:
        json.dump(split_dataset, new_file, ensure_ascii=False, indent=2)

  # Score the subsolutions using the configured model.
  subscoring_conversations = make_subscoring_conversations(split_dataset, SUBSOLUTION_EXTRACTION_PROMPT)
  responses = llm.chat(messages=subscoring_conversations, sampling_params=sampling_params, use_tqdm=True)
  scored_dataset = score_subsolutions(split_dataset, responses)
  if args.checkpoint:
    outfile = os.path.join(args.result_dir, f"scored-subsolutions.json")
    with open(outfile, 'w', encoding='utf-8') as new_file:
        json.dump(scored_dataset, new_file, ensure_ascii=False, indent=2)

  # Rewrite response based on variants of combining sub-solutions. Here are examples for
  # FCS, FCS+1, and FCS+Reflection. 
  variants_dataset = build_response_variants(scored_dataset)
  if args.checkpoint:
    outfile = os.path.join(args.result_dir, f"response-variants.json")
    with open(outfile, 'w', encoding='utf-8') as new_file:
        json.dump(variants_dataset, new_file, ensure_ascii=False, indent=2)

  # Add per-variant token counts to dataset for convenience.
  final_dataset = compute_token_usages(variants_dataset, llm)

  system_prompt = SYSTEM_PROMPT[args.target_model]

  # Save each variant in conversation format.
  fcs_convo = make_preference_conversations(final_dataset, "fcs", system_prompt)
  fcs_outfile = os.path.join(args.result_dir, "fcs-conversations.json")
  with open(fcs_outfile, 'w', encoding='utf-8') as new_file:
    json.dump(fcs_convo, new_file, ensure_ascii=False, indent=2)
        
  fcs_plus1_convo = make_preference_conversations(final_dataset, "fcs_plus1", system_prompt)
  fcs_plus1_outfile = os.path.join(args.result_dir, "fcs_plus1-conversations.json")
  with open(fcs_plus1_outfile, 'w', encoding='utf-8') as new_file:
    json.dump(fcs_plus1_convo, new_file, ensure_ascii=False, indent=2)
        
  fcs_reflection_convo = make_preference_conversations(final_dataset, "fcs_reflection", system_prompt)
  fcs_reflection_outfile = os.path.join(args.result_dir, "fcs_reflection-conversations.json")
  with open(fcs_reflection_outfile, 'w', encoding='utf-8') as new_file:
    json.dump(fcs_reflection_convo, new_file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
