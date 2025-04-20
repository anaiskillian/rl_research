import time
import vllm
from vllm import SamplingParams
from datasets import load_dataset

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

def run_experiment(cot_flag: bool):
    # Load the GSM8k dataset ("gsm8k" with "main" config, using test split for a quick experiment)
    dataset = load_dataset("gsm8k", "main", split="test")
    sample = dataset[0]  # using the first sample as a minimal example
    problem = sample["question"]
    ground_truth = sample["answer"]

    # Create the initial prompt.
    prompt = render_prompt(problem, ground_truth, cot=cot_flag)
    print("==== Prompt ====")
    print(prompt)

    # Instantiate the vLLM model.
    llm = vllm.LLM("Qwen/Qwen2.5-0.5B-Instruct")

    # Define simple sampling parameters.
    # Here, temperature=0.0 and top_p=1.0 for deterministic generation.
    # Setting logprobs=0 and prompt_logprobs=0 indicates returning only the sampled token’s logprob.
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, logprobs=0, prompt_logprobs=0)
    
    # First turn: generate model output for the problem.
    t_start = time.time()
    outputs = llm.generate([prompt], sampling_params)
    t_end = time.time()
    first_output = outputs[0].outputs[0].text
    print("==== First Turn Output ====")
    print(first_output)
    print(f"First generation took {t_end - t_start:.2f} seconds")

    # Second turn: append the known ground truth after the generated text
    # to force the model to consider the correct answer, enabling us to measure the log likelihood.
    second_prompt = f"{prompt}{first_output}\nThe correct answer is: {ground_truth}"
    print("\n==== Second Prompt ====")
    print(second_prompt)

    # Generate second turn output.
    t_start = time.time()
    outputs2 = llm.generate([second_prompt], sampling_params)
    t_end = time.time()
    second_output = outputs2[0].outputs[0].text
    print("==== Second Turn Output ====")
    print(second_output)
    print(f"Second generation took {t_end - t_start:.2f} seconds")

    # Extract and print log probabilities for the ground truth tokens.
    logprobs = get_logprobs(outputs2[0], ground_truth)
    print("==== Log Probability Data ====")
    print(logprobs)

def main():
    print("=== Experiment WITHOUT Chain-of-Thought ===")
    run_experiment(cot_flag=False)
    
    print("\n=== Experiment WITH Chain-of-Thought ===")
    run_experiment(cot_flag=True)

if __name__ == "__main__":
    main()
