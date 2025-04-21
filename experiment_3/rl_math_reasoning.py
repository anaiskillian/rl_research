import time
import vllm
import re
from vllm import SamplingParams
from datasets import load_dataset
from typing import Tuple, Dict, List

def render_prompt(problem: str, ground_truth: str, cot: bool = False) -> str:
    """
    Creates a prompt for the math problem.
    If cot is True, includes a chain-of-thought instruction.
    """
    base_instruction = "Solve the following math problem:"
    cot_instruction = "Think step-by-step before giving your final answer." if cot else "Provide the final answer directly."
    prompt = (
        f"{base_instruction}\n"
        f"{cot_instruction}\n\n"
        f"Problem:\n{problem}\n\n"
        f"Solution:\n"
    )
    return prompt

def get_logprobs(model_output, ground_truth: str):
    """
    Extracts token-level log probabilities from the model output.
    For simplicity, we print the 'prompt_logprobs' field.
    In a more advanced version you would tokenize ground_truth and extract corresponding logprobs.
    """
    try:
        # Here model_output.prompt_logprobs is a list or dictionary provided by vLLM.
        # You can further process it to compute average log likelihood etc.
        prompt_logprobs = model_output.prompt_logprobs
    except Exception as e:
        print("Error extracting logprobs:", e)
        prompt_logprobs = None
    return prompt_logprobs

def extract_final_answer(text: str) -> str:
    """
    Extracts the final numerical answer from the model's output.
    Returns cleaned answer string or None if no answer found.
    """
    # First try to find LaTeX boxed answer
    boxed_pattern = r"\\boxed{([^}]+)}"
    boxed_match = re.search(boxed_pattern, text)
    if boxed_match:
        return boxed_match.group(1).strip()
    
    # If no boxed answer found, look for other patterns
    patterns = [
        r"(?:final answer|answer is|=)\s*(-?\d+(?:\.\d+)?)",  # matches "answer is 42" or "= 42"
        r"(-?\d+(?:\.\d+)?)\s*$",  # matches number at the end of string
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            return match.group(1).strip()
    return None

def check_answer_correctness(predicted: str, ground_truth: str) -> Tuple[bool, str]:
    """
    Checks if the predicted answer matches the ground truth.
    Returns (is_correct, error_message).
    """
    if predicted is None:
        return False, "No numerical answer found in output"
    
    try:
        # Clean and convert both to floats for numerical comparison
        pred_num = float(predicted.replace(',', ''))
        # Extract number from ground truth (assuming format like "The answer is 42")
        truth_match = re.search(r"(-?\d+(?:\.\d+)?)", ground_truth)
        if not truth_match:
            return False, "Could not extract number from ground truth"
        
        truth_num = float(truth_match.group(1))
        
        # Check if numbers are equal (with small tolerance for floating point)
        is_correct = abs(pred_num - truth_num) < 1e-6
        return is_correct, "" if is_correct else f"Expected {truth_num}, got {pred_num}"
    
    except ValueError:
        return False, "Error converting answers to numbers"

def run_sanity_checks(model_output: str, problem: str) -> List[str]:
    """
    Performs basic sanity checks on the model output.
    Returns list of warning messages.
    """
    warnings = []
    
    # Check if output is not empty
    if not model_output or model_output.isspace():
        warnings.append("Empty or whitespace-only output")
        
    # Check if output is not too short
    if len(model_output) < 10:
        warnings.append("Suspiciously short output")
        
    # Check if output contains numerical digits
    if not any(c.isdigit() for c in model_output):
        warnings.append("No numerical digits in output")
        
    # Check if output is not just repeating the problem
    if problem.strip() in model_output.strip():
        warnings.append("Output contains exact copy of input problem")
    
    return warnings

def run_experiment(cot_flag: bool, num_samples: int = 50):
    """
    Runs the experiment with accuracy tracking and sanity checks.
    Args:
        cot_flag: Whether to use chain-of-thought prompting
        num_samples: Number of samples to evaluate (default: 50)
    """
    # Load the GSM8k dataset
    dataset = load_dataset("gsm8k", "main", split="test")
    if num_samples > len(dataset):
        num_samples = len(dataset)
    
    # Initialize tracking variables
    correct_count = 0
    total_time = 0
    all_warnings = []
    results = []

    # Instantiate the vLLM model with proper Qwen configuration
    llm = vllm.LLM(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        trust_remote_code=True,  # Required for Qwen
        tensor_parallel_size=1,  # Adjust based on your GPU setup
        gpu_memory_utilization=0.9,
        max_model_len=2048
    )
    
    # Configure sampling parameters
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["####"]  # Stop at the answer marker
    )

    print(f"Running experiment {'with' if cot_flag else 'without'} Chain-of-Thought")
    print(f"Evaluating {num_samples} samples...")

    for i in range(num_samples):
        sample = dataset[i]
        problem = sample["question"]
        ground_truth = sample["answer"]
        
        # Create prompt and generate response
        prompt = render_prompt(problem, ground_truth, cot=cot_flag)
        t_start = time.time()
        outputs = llm.generate([prompt], sampling_params)
        t_end = time.time()
        
        generation_time = t_end - t_start
        total_time += generation_time
        
        # Get model output and extract answer
        model_output = outputs[0].outputs[0].text
        predicted_answer = extract_final_answer(model_output)
        true_answer = extract_final_answer(ground_truth)
        
        # Check correctness
        is_correct = predicted_answer == true_answer if predicted_answer and true_answer else False
            
        # Run sanity checks
        warnings = run_sanity_checks(model_output, problem)
        if warnings:
            all_warnings.extend([f"Sample {i+1}: {w}" for w in warnings])
            
        # Store result
        results.append({
            "sample_id": i,
            "problem": problem,
            "model_response": model_output,
            "predicted_answer": predicted_answer,
            "ground_truth": true_answer,
            "correct": is_correct,
            "time": generation_time,
            "warnings": warnings
        })
        
        # Print progress
        if (i + 1) % 10 == 0:
            print(f"Processed {i+1}/{num_samples} samples. Current accuracy: {(correct_count/(i+1))*100:.2f}%")

    # Print final results
    print("\n=== Final Results ===")
    print(f"Total samples: {num_samples}")
    print(f"Correct answers: {correct_count}")
    print(f"Accuracy: {(correct_count/num_samples)*100:.2f}%")
    print(f"Average time per sample: {total_time/num_samples:.2f} seconds")
    
    if all_warnings:
        print("\n=== Warnings ===")
        for warning in all_warnings:
            print(warning)

    return results

def main():
    print("=== Experiment WITHOUT Chain-of-Thought ===")
    results_no_cot = run_experiment(cot_flag=False, num_samples=50)
    
    print("\n=== Experiment WITH Chain-of-Thought ===")
    results_cot = run_experiment(cot_flag=True, num_samples=50)
    
    # Compare results
    print("\n=== Comparison ===")
    acc_no_cot = sum(r["correct"] for r in results_no_cot) / len(results_no_cot)
    acc_cot = sum(r["correct"] for r in results_cot) / len(results_cot)
    time_no_cot = sum(r["time"] for r in results_no_cot) / len(results_no_cot)
    time_cot = sum(r["time"] for r in results_cot) / len(results_cot)
    
    print(f"No CoT Accuracy: {acc_no_cot*100:.2f}%")
    print(f"With CoT Accuracy: {acc_cot*100:.2f}%")
    print(f"No CoT Average Time: {time_no_cot:.2f}s")
    print(f"With CoT Average Time: {time_cot:.2f}s")

if __name__ == "__main__":
    main()
