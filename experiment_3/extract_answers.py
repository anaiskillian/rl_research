import csv
import json
import re
import vllm
import os
from vllm import SamplingParams
from datetime import datetime

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
        r"\\boxed{([^}]+)}",  # prioritize boxed
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
    "Solve the following math problem step by step. "
    "Show your work and reasoning clearly. "
    "At the end, it is required to PROVIDE the calculated answer in a box like this: \\boxed{your_answer}."
)

NOCOT_PROMPT = (
    "Do NOT think about the problem step-by-step."
    "YOU MUST PROVIDE only your FINAL answer boxed like this: \\boxed{your_answer}."
)

def process_csv_file(filepath, llm, sampling_params, mode, limit=5):
    """Process a single CSV file and return parsed results."""
    results = []
    print(f"\nProcessing {mode.upper()} file:")
    
    # Check if file exists
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} not found.")
        return results
        
    try:
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= limit:
                    break

                problem = row.get('problem', '')
                ground_truth = row.get('ground_truth_answer', '')
                
                if not problem:
                    print(f"Warning: Row {i+1} missing problem field, skipping.")
                    continue

                if mode == "cot":
                    prompt = f"{COT_PROMPT}\nProblem: {problem}\nAnswer:"
                else:  # nocot
                    prompt = f"{NOCOT_PROMPT}\nProblem: {problem}\nAnswer:"

                response = generate_response(llm, prompt, sampling_params)
                model_answer = extract_model_answer(response)
                true_answer = extract_answer_from_ground_truth(ground_truth)

                if model_answer is None:
                    print(f"[Warning] Could not extract model answer for:\n{response}")

                print(f"\nRow {i + 1}:")
                print(f"Problem: {problem}")
                print(f"Model response: {response}")
                print(f"Model answer: {model_answer}")
                print(f"Ground truth answer: {true_answer}")

                results.append({
                    'problem_idx': row.get('problem_idx', str(i)),
                    'problem': problem,
                    'model_response': response,
                    'model_answer': model_answer,
                    'ground_truth': true_answer
                })
    except Exception as e:
        print(f"Error processing file {filepath}: {str(e)}")
        
    return results

def process_files():
    # Create output directory and get timestamp
    output_dir = ensure_output_dir()
    timestamp = get_timestamp()
    
    # Model configuration
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
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
    llm = vllm.LLM(
        model=model_name,
        trust_remote_code=model_info["parameters"]["trust_remote_code"],
        tensor_parallel_size=model_info["parameters"]["tensor_parallel_size"],
        gpu_memory_utilization=model_info["parameters"]["gpu_memory_utilization"],
        max_model_len=model_info["parameters"]["max_model_len"]
    )

    # Sampling configuration
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=1024,
    )

    # File paths
    cot_path = '/n/netscratch/kdbrantley_lab/Lab/akillian/results2/cot/results_shot2_cot.csv'
    nocot_path = '/n/netscratch/kdbrantley_lab/Lab/akillian/results2/nocot/results_shot2_nocot.csv'

    # Process both modes
    results = {
        'cot': process_csv_file(cot_path, llm, sampling_params, mode='cot'),
        'nocot': process_csv_file(nocot_path, llm, sampling_params, mode='nocot')
    }

    # Save experiment info and results
    save_experiment_info(output_dir, timestamp, COT_PROMPT, NOCOT_PROMPT, results, model_info)

if __name__ == '__main__':
    process_files()