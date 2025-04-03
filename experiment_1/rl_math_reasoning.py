import os
import re
import json
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import sys

import hydra
from omegaconf import DictConfig
import pandas as pd
import numpy as np
from tqdm import tqdm
from datasets import load_dataset

import vllm
from vllm import SamplingParams

from consts import COLUMNS_TO_SAVE


@dataclass
class EvalRequest:
    subject: str
    problem_idx: int
    problem: str
    solution: str
    examples: List[Tuple[str, str]]
    cot: bool

@dataclass
class Turn:
    prompt: str
    response: Optional[str] = None
    logprobs: Optional[Dict] = None
    perplexity: Optional[float] = None
    tokens: Optional[List[str]] = None

@dataclass
class Turns:
    solution_turn: Turn  # Model's attempt at solution
    ground_truth_turn: Turn  # Turn for measuring perplexity on ground truth


def get_problems_and_solutions_from_dataset(
    dataset, 
    include_ground_truth_explanation_in_second_prompt: bool = True, 
    keep_boxes_in_answer: bool = True
):
    problems = {}
    solutions = {}
    sorted_indices = list(range(len(dataset)))
    
    for idx, item in enumerate(dataset):
        problems[idx] = item["question"]
        if include_ground_truth_explanation_in_second_prompt:
            solutions[idx] = item["answer"]
        else:
            solutions[idx] = extract_final_answer(item["answer"], keep_boxes_in_answer)

    return problems, solutions, sorted_indices


def cot_text():
    return "Think about the following problem step-by-step (explicitly write out each step before arriving at your final answer).\n"

def no_cot_text():
    return "Do not think about the problem step-by-step. Write out your final answer directly.\n"

def problem_instructions():
    return "Remember to end your solution with the text '#### ' followed by your final answer.\n"

def explanation_instructions(keep_boxes_in_answer: bool = True):
    if keep_boxes_in_answer:
        return "For your explanation, start with 'Explanation:' before giving your final boxed answer.\n"
    else:
        return "For your explanation, start with 'Explanation:' before giving your final answer.\n"

def sys_prompt():
    # A system prompt that tells the model its role
    return (
        "<|im_start|>system\n<|im_end|>\n"
        "<|im_start|>user\nYou are a helpful mathematics assistant<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

def sys_end_prompt():
    return "<|im_end|><|im_start|>assistant"

def extract_final_answer(solution: str, keep_boxes_in_answer: bool = True) -> str:
    matches = []
    level = 0
    start = -1
    
    boxed_start = solution.rfind('\\boxed{')
    if boxed_start == -1:
        if keep_boxes_in_answer:
            return "$\\boxed{" + solution + "}$"
        else:
            return "$" + solution + "$"
    
    i = boxed_start + 7  # position after \boxed{
    
    while i < len(solution):
        if solution[i] == '{':
            level += 1
            if level == 1:
                start = i
        elif solution[i] == '}':
            level -= 1
            if level == -1:
                if keep_boxes_in_answer:
                    return "$\\boxed{" + solution[boxed_start+7:i] + "}$"
                else:
                    return "$" + solution[boxed_start+7:i] + "$"
        i += 1
    if keep_boxes_in_answer:
        return "$\\boxed{" + solution + "}$"
    else:
        return "$" + solution + "$"

def render_prompt(problem: str, examples: List[Tuple[str, str]] = None, cot: bool = False, keep_boxes_in_answer: bool = True) -> str:
    prompt_parts = [sys_prompt()]
    if examples:
        prompt_parts.append("Consider the following examples of math problems and their solutions:\n\n")
        for i, (example_problem, example_solution) in enumerate(examples, 1):
            prompt_parts.append(f"Example {i}:\n")
            prompt_parts.append("Problem:\n")
            prompt_parts.append(example_problem + "\n\n")
            prompt_parts.append("Explanation:\n")
            answer = extract_final_answer(example_solution, keep_boxes_in_answer)
            prompt_parts.append(f"{example_solution}\n#### {answer}\n\n")
    
    prompt_parts.append("Now, solve the following problem.\n\n")
    if cot:
        prompt_parts.append(cot_text())
        prompt_parts.append(explanation_instructions(keep_boxes_in_answer))
    else:
        prompt_parts.append(no_cot_text())
    
    prompt_parts.append(problem_instructions() + "\n")
    prompt_parts.append("Problem:\n")
    prompt_parts.append(problem + "\n\n")
    prompt_parts.append("Solution:\n")

    prompt_parts.append(sys_end_prompt())
    
    return "".join(prompt_parts)


def calculate_perplexity(logprobs: List[float]) -> float:
    if not logprobs:
        return float('inf')
    avg_log_prob = sum(logprobs) / len(logprobs)
    return np.exp(-avg_log_prob)

def get_answer_logprobs(model_response: Dict, ground_truth: str) -> Dict:
    """Extract log probabilities for the ground truth answer tokens."""
    if not model_response.get('choices'):
        return {
            'answer_logprobs': None,
            'answer_perplexity': None,
            'answer_tokens': None
        }
    
    logprobs = model_response['choices'][0].get('logprobs', [])
    if not logprobs:
        return {
            'answer_logprobs': None,
            'answer_perplexity': None,
            'answer_tokens': None
        }

    tokens = logprobs['tokens']
    token_logprobs = logprobs['token_logprobs']

    # Attempt to match the ground truth against generated tokens
    decoded_answer = ''.join(tokens).strip()
    if ground_truth.strip() not in decoded_answer:
        return {
            'answer_logprobs': None,
            'answer_perplexity': None,
            'answer_tokens': None
        }

    return {
        'answer_logprobs': token_logprobs,
        'answer_perplexity': calculate_perplexity(token_logprobs),
        'answer_tokens': tokens
    }


def process_batch_evaluation(config: DictConfig, batch_requests: List[EvalRequest], llm) -> List[dict]:
    """Process a batch of evaluation requests using vLLM."""
    # Prepare first turn prompts for the batch
    first_prompts = [
        render_prompt(
            problem=req.problem,
            examples=req.examples,
            cot=req.cot,
            keep_boxes_in_answer=config.keep_boxes_in_answer
        ) for req in batch_requests
    ]
    
    sampling_params = SamplingParams(temperature=config.temperature, top_p=config.top_p, logprobs=0, prompt_logprobs=0)
    
    t1 = time.time()
    print(f"First prompt size: {len(json.dumps(first_prompts))/1024:.1f} KB")
    
    first_outputs = llm.generate(first_prompts, sampling_params)
    t2 = time.time()
    print(f"First generation took {t2 - t1:.2f} seconds")
    
    second_prompts = []
    for req, first_output in zip(batch_requests, first_outputs):
        ground_truth = req.solution.strip()
        first_prompt = render_prompt(req.problem, req.examples, req.cot, keep_boxes_in_answer=config.keep_boxes_in_answer)
        
        if req.cot:
            if config.remove_model_answer_in_second_prompt:
                reasoning = first_output.outputs[0].text.split('####')[0] if '####' in first_output.outputs[0].text else ''
            else:
                reasoning = first_output.outputs[0].text
            second_prompt = f"{first_prompt}{reasoning}\nThe correct answer is: {ground_truth}<|eot_id|>"
        else:
            second_prompt = f"{first_prompt}{first_output.outputs[0].text}\nThe correct answer is: {ground_truth}<|eot_id|>"
        second_prompts.append(second_prompt)
    
    t1 = time.time()
    print(f"Second prompt size: {len(json.dumps(second_prompts))/1024:.1f} KB")
    second_outputs = llm.generate(second_prompts, sampling_params)[0]
    t2 = time.time()
    print(f"Second generation took {t2 - t1:.2f} seconds")
    
    results = []
    for i, (req, first_output, second_output) in enumerate(zip(batch_requests, first_outputs, second_outputs)):
        logprobs_data = get_answer_logprobs({"choices": [second_output]}, req.solution.strip())
        explanation = ""
        if req.cot:
            match_explanation = re.search(config.explanation_regex, first_output.outputs[0].text, re.DOTALL)
            explanation = match_explanation.group('explanation').strip() if match_explanation else ""
        
        result = {
            'subject': req.subject,
            'problem_idx': req.problem_idx,
            'problem': req.problem,
            'ground_truth_answer': req.solution,
            'first_turn_prompt': first_prompts[i],
            'first_turn_response': first_output.outputs[0].text,
            'second_turn_prompt': second_prompts[i],
            'second_turn_response': second_output.outputs[0].text,
            'explanation': explanation if req.cot else None,
            'perplexity': logprobs_data.get('answer_perplexity'),
            'ground_truth_logprobs': logprobs_data.get('answer_logprobs'),
            'ground_truth_tokens': logprobs_data.get('answer_tokens'),
            'first_generation_logprobs': first_output.get('logprobs'),
            'second_generation_logprobs': second_output.get('logprobs'),
            'first_prompt_logprobs': first_output.get('prompt_logprobs'),
            'second_prompt_logprobs': second_output.get('prompt_logprobs'),
        }
        results.append(result)
        print(f"Completed evaluation for subject {req.subject}, problem {req.problem_idx}")
        print("Result:", result)  # <-- New print added here
    
    return results


@hydra.main(config_path=".", config_name="math_config_instruct")
def main(cfg: DictConfig):
    # Redirect prints to a log file in the specified output directory.
    os.makedirs(cfg.output_path, exist_ok=True)
    log_file_path = os.path.join(cfg.output_path, "run_log.txt")
    sys.stdout = open(log_file_path, "w")
    
    BATCH_SIZE = cfg.batch_size
    shot = cfg.shot
    cot = cfg.cot

    print(f"Number of shots: {shot}")
    print(f"Chain of Thought enabled: {cot}")
    print(f"Perplexity measurement enabled: {cfg.get('measure_perplexity', False)}")
    print(f"Batch size: {BATCH_SIZE}")
    print("-" * 50)

    # Load the dataset using the Hugging Face datasets library.
    hf_dataset = load_dataset(cfg.dataset_name, "main")[cfg.split_to_use]
    problems, solutions, sorted_indices = get_problems_and_solutions_from_dataset(
        hf_dataset,
        include_ground_truth_explanation_in_second_prompt=True,
        keep_boxes_in_answer=cfg.keep_boxes_in_answer
    )
    shot_indices = sorted_indices[:shot]
    examples = [(problems[i], solutions[i]) for i in shot_indices] if shot > 0 else []

    eval_requests = []
    eval_indices = sorted_indices[shot:]
    for idx in eval_indices:
        eval_requests.append(EvalRequest(
            subject="math",
            problem_idx=idx,
            problem=problems[idx],
            solution=solutions[idx],
            examples=examples.copy() if examples else None,
            cot=cot
        ))
    
    print(f"Total evaluation requests: {len(eval_requests)}")
    
    # Instantiate the vLLM model with Qwen/Qwen2.5-0.5B-Instruct.
    llm = vllm.LLM("Qwen/Qwen2.5-0.5B-Instruct")
    
    results = []
    num_batches = (len(eval_requests) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min((batch_idx + 1) * BATCH_SIZE, len(eval_requests))
        batch = eval_requests[start_idx:end_idx]
        
        print(f"\nProcessing batch {batch_idx + 1}/{num_batches}")
        batch_results = process_batch_evaluation(cfg, batch, llm)
        results.extend(batch_results)
        print(f"Completed batch {batch_idx + 1}/{num_batches}")
    
    df = pd.DataFrame(results)
    csv_path = os.path.join(
        cfg.output_path, 
        f"results_shot{shot}_{'cot' if cot else 'nocot'}.csv"
    )
    df[COLUMNS_TO_SAVE].to_csv(csv_path, index=False)
    
    print("CSV file saved to:", csv_path)
    print("Results:")
    print(df[COLUMNS_TO_SAVE])
    
    if cfg.get('measure_perplexity', False):
        overall_perplexity = df['perplexity'].mean()
        print(f"Overall perplexity: {overall_perplexity:.4f}")

if __name__ == "__main__":
    main()
