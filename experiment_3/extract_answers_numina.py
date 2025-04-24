import csv
import json
import re
import vllm
import os
from vllm import SamplingParams
from datetime import datetime
import datasets

def ensure_output_dir():
    """Create output directory if it doesn't exist."""
    output_dir = "math_experiment_outputs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def get_timestamp():
    """Get current timestamp in a readable format."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def save_experiment_info(output_dir, timestamp, cot_prompt, nocot_prompt, results, model_info):
    """Save experiment information and results to a timestamped file."""
    filename = f"experiment_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    experiment_data = {
        "timestamp": timestamp,
        "model": model_info,
        "prompts": {
            "cot": cot_prompt,
            "nocot": nocot_prompt
        },
        "results": results
    }
    
    with open(filepath, 'w') as f:
        json.dump(experiment_data, f, indent=2)
    
    print(f"\nResults saved to: {filepath}")
    return filepath

def extract_answer_from_ground_truth(text):
    """Extract the final answer from the ground truth text."""
    if not isinstance(text, str):
        return None

    # First try \boxed{}
    match = re.search(r"\\boxed{([^}]+)}", text)
    if match:
        return match.group(1).strip()
    
    # Try finding a numeric answer marked with ####
    match = re.search(r"####\s*([^#\n]+)", text)
    if match:
        return match.group(1).strip()

    # Fallback: try extracting any final number (as was likely before)
    match = re.search(r"(-?\d+(?:\.\d+)?)\s*$", text)
    if match:
        return match.group(1).strip()

    # Final fallback: maybe it's just a raw number already
    return text.strip()

def extract_model_answer(response):
    """Extract the final answer from the model's response."""
    if not isinstance(response, str):
        return None
    response = response.strip()

    patterns = [
        r"\\boxed{([^}]*)",  # prioritize boxed, without requiring closing brace
        r"####\s*(-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)",  # markdown style
        r"(?:final answer|answer is|=)\s*(-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)",
        r"(-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)(?!.*\d)"  # fallback: match LAST number
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).replace(',', '').strip()

    return None

def normalize_newlines(text: str) -> str:
    """Normalize newlines by collapsing 3+ into 2."""
    return re.sub(r'\n{3,}', '\n\n', text.strip())

def generate_response(llm, prompt, sampling_params):
    """Generate a response from the model."""
    outputs = llm.generate([prompt], sampling_params)
    return normalize_newlines(outputs[0].outputs[0].text.strip())

# Define prompts at the top level
COT_PROMPT = (
    "Solve the following math problem step by step. At the end, it is required to PROVIDE the calculated answer in a box like this: \\boxed{your_answer}."
)

NOCOT_PROMPT = (
    "Do NOT think about the problem step-by-step. You must provide ONLY your FINAL answer boxed like this: \\boxed{your_answer}."
)

def filter_word_problems(problems, limit=100):
    """Filter out only the math word problems with numeric answers."""
    filtered = []
    for problem in problems:
        if problem.get('question_type') == 'math-word-problem' and problem.get('answer') and problem.get('answer') != 'proof':
            filtered.append(problem)
            if len(filtered) >= limit:
                break
    return filtered

def load_numina_math():
    """Load NuminaMath-1.5 dataset from Hugging Face."""
    print("Loading NuminaMath-1.5 dataset from Hugging Face...")
    dataset = datasets.load_dataset("AI-MO/NuminaMath-1.5", split="train")
    return dataset

def convert_to_experiment_format(problems, num_samples=5):
    """Convert NuminaMath problems to our experiment format."""
    results = []
    print(f"Converting {num_samples} NuminaMath problems to the required format...")
    
    for i, item in enumerate(problems):
        if i >= num_samples:
            break
            
        # Get problem and solution from dataset
        problem = item.get('problem', '')
        solution = item.get('solution', '')
        answer = item.get('answer', '')
        
        if problem and (solution or answer):
            results.append({
                'problem_idx': str(i),
                'problem': problem,
                'ground_truth_answer': answer,
                'solution': solution  # Keep the full solution for reference
            })
    
    return results

def process_problems(problems, llm, sampling_params, mode, limit=5):
    """Process problems and return parsed results."""
    results = []
    print(f"\nProcessing {mode.upper()} problems:")
    
    for i, problem_data in enumerate(problems):
        if i >= limit:
            break

        problem = problem_data.get('problem', '')
        ground_truth = problem_data.get('ground_truth_answer', '')
        
        if not problem:
            print(f"Warning: Problem {i+1} missing problem text, skipping.")
            continue

        if mode == "cot":
            prompt = f"{COT_PROMPT}\nProblem: {problem}\nAnswer:"
        else:  # nocot
            prompt = f"{NOCOT_PROMPT}\nProblem: {problem}\nAnswer:"

        response = generate_response(llm, prompt, sampling_params)
        model_answer = extract_model_answer(response)
        
        if model_answer is None:
            print(f"[Warning] Could not extract model answer for:\n{response}")

        print(f"\nProblem {i + 1}:")
        print(f"Problem: {problem}")
        print(f"Model response: {response}")
        print(f"Model answer: {model_answer}")
        print(f"Ground truth answer: {ground_truth}")

        results.append({
            'problem_idx': problem_data.get('problem_idx', str(i)),
            'problem': problem,
            'model_response': response,
            'model_answer': model_answer,
            'ground_truth': ground_truth
        })
    
    return results

def process_files(num_samples=5):
    # Create output directory and get timestamp
    output_dir = ensure_output_dir()
    timestamp = get_timestamp()
    
    # Model configuration
    model_name = "Qwen/Qwen2.5-3B-Instruct"
    model_info = {
        "name": model_name,
        "parameters": {
            "trust_remote_code": True,
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.9,
            "max_model_len": 2048
        }
    }
    
    # Initialize model
    print(f"Initializing {model_name}...")
    llm = vllm.LLM(
        model=model_name,
        trust_remote_code=model_info["parameters"]["trust_remote_code"],
        tensor_parallel_size=model_info["parameters"]["tensor_parallel_size"],
        gpu_memory_utilization=model_info["parameters"]["gpu_memory_utilization"],
        max_model_len=model_info["parameters"]["max_model_len"]
    )

    # Updated sampling configuration with stop sequences
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=512,
        stop=["}"]
    )

    # Load NuminaMath-1.5 dataset directly from Hugging Face
    dataset = load_numina_math()
    
    # Filter to only include word problems with numeric answers
    filtered_problems = filter_word_problems(dataset, limit=100)
    
    # Convert to our experiment format
    test_problems = convert_to_experiment_format(filtered_problems, num_samples=num_samples)
    
    print(f"Loaded {len(test_problems)} problems from NuminaMath-1.5 dataset")
    
    # Process both modes using NuminaMath data
    results = {
        'cot': process_problems(test_problems, llm, sampling_params, mode='cot', limit=num_samples),
        'nocot': process_problems(test_problems, llm, sampling_params, mode='nocot', limit=num_samples)
    }

    # Save experiment info and results
    save_experiment_info(output_dir, timestamp, COT_PROMPT, NOCOT_PROMPT, results, model_info)
    
    # Save problems to CSV for reference
    csv_path = os.path.join(output_dir, f"numina_math_problems_{timestamp}.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['problem_idx', 'problem', 'ground_truth_answer']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for problem in test_problems:
            writer.writerow({
                'problem_idx': problem['problem_idx'],
                'problem': problem['problem'],
                'ground_truth_answer': problem['ground_truth_answer']
            })
    
    print(f"Problems saved to: {csv_path}")

if __name__ == '__main__':
    # You can adjust the number of samples to evaluate here
    process_files(num_samples=5)