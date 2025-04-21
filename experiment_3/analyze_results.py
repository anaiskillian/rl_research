import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import re
import sympy
from sympy.parsing.latex import parse_latex
import signal
from typing import Tuple, List
from tabulate import tabulate

# ---------------------------
# 1. Load CSVs and parse logprobs
# ---------------------------
CSV_PATHS = {
    "cot": "/n/netscratch/kdbrantley_lab/Lab/akillian/results2/results_shot2_cot.csv",
    "nocot": "/n/netscratch/kdbrantley_lab/Lab/akillian/results_shot2_nocot.csv",
}

TO_ANALYZE = {}
for key in CSV_PATHS:
    TO_ANALYZE[key] = pd.read_csv(CSV_PATHS[key])

# Drop rows where the logprobs column is missing
TO_ANALYZE["cot"] = TO_ANALYZE["cot"].dropna(subset=["second_prompt_logprobs"])
TO_ANALYZE["nocot"] = TO_ANALYZE["nocot"].dropna(subset=["second_prompt_logprobs"])

# Parse the stored (stringified) logprobs column using ast.literal_eval
TO_ANALYZE["cot"]["second_prompt_logprobs_parsed"] = TO_ANALYZE["cot"]["second_prompt_logprobs"].apply(ast.literal_eval)
TO_ANALYZE["nocot"]["second_prompt_logprobs_parsed"] = TO_ANALYZE["nocot"]["second_prompt_logprobs"].apply(ast.literal_eval)

# ---------------------------
# 2. Functions to calculate perplexity and extract answer token logprobs
# ---------------------------
def calculate_perplexity(logprobs: List[float]) -> float:
    if len(logprobs) == 0:
        return float('inf')
    avg_log_prob = sum(logprobs) / len(logprobs)
    return np.exp(-avg_log_prob)

def find_logprobs_on_answer(logprobs) -> Tuple[List[float], List[str], float]:
    """
    Traverse the list of logprobs backwards until a marker is found (here we use
    the appearance of '$\\boxed{' as an example).
    Returns a tuple of (list of logprobs for answer tokens, list of corresponding tokens, computed perplexity).
    """
    logprobs_answer = []
    tok_answer = []
    should_break = False
    for idx in range(len(logprobs)-1, -1, -1):
        if not logprobs[idx]:
            continue
        for key in logprobs[idx]:
            token_info = logprobs[idx][key]
            # Check if the token (after stripping) matches our starting marker.
            if token_info['decoded_token'].strip() == r'$\boxed{':
                should_break = True
            tok_answer.append(token_info['decoded_token'])
            logprobs_answer.append(token_info['logprob'])
        if should_break:
            break
    # Reverse lists so tokens appear in original order
    logprobs_answer = logprobs_answer[::-1]
    tok_answer = tok_answer[::-1]
    ppl = calculate_perplexity(logprobs_answer)
    return logprobs_answer, tok_answer, ppl

# For example, compute perplexity for the first sample (if available)
sample_cot = TO_ANALYZE["cot"]["second_prompt_logprobs_parsed"].iloc[0]
lp_sample, toks_sample, ppl_sample = find_logprobs_on_answer(sample_cot)
print("Sample tokens:", ''.join(toks_sample))
print("Sample perplexity =", ppl_sample)

# ---------------------------
# 3. Plot histograms of perplexity
# ---------------------------
hist_cot_ppl = [find_logprobs_on_answer(TO_ANALYZE["cot"]["second_prompt_logprobs_parsed"].iloc[i])[2] 
                for i in range(len(TO_ANALYZE["cot"]))]
hist_nocot_ppl = [find_logprobs_on_answer(TO_ANALYZE["nocot"]["second_prompt_logprobs_parsed"].iloc[i])[2] 
                  for i in range(len(TO_ANALYZE["nocot"]))]

plt.figure(figsize=(10, 5))
plt.hist(hist_cot_ppl, bins=30, histtype='step', linewidth=2, label='COT', alpha=0.5)
plt.hist(hist_nocot_ppl, bins=30, histtype='step', linewidth=2, label='No COT', alpha=0.5)
plt.xlabel("Perplexity (GT answer)")
plt.ylabel("Frequency")
plt.yscale('log')
plt.legend(loc='upper right')
plt.title("Histogram of Perplexities")
plt.show()

# ---------------------------
# 4. Functions for answer extraction and evaluation
# ---------------------------
def remove_pound_signs(text: str) -> str:
    return text.replace("#", "")

def extract_final_answer(response: str, keep_boxes_in_answer: bool = False) -> str:
    """
    Extract the portion after '####' and then, if present, extract the content inside \boxed{...}.
    """
    parts = response.split('####')
    if len(parts) > 1:
        answer = parts[1].strip()
    else:
        answer = response.strip()
    idx = answer.find(r'\boxed')
    if idx == -1:
        if '$' not in answer:
            return f"${answer}$"
        else:
            return answer
    i = idx + len(r'\boxed')
    while i < len(answer) and answer[i].isspace():
        i += 1
    if i >= len(answer) or answer[i] != '{':
        if '$' not in answer:
            return f"${answer}$"
        else:
            return answer
    i += 1  # skip the '{'
    brace_level = 1
    contents = []
    while i < len(answer) and brace_level > 0:
        c = answer[i]
        if c == '{':
            brace_level += 1
            contents.append(c)
        elif c == '}':
            brace_level -= 1
            if brace_level > 0:
                contents.append(c)
        else:
            contents.append(c)
        i += 1
    inside_box = ''.join(contents).strip()
    if keep_boxes_in_answer:
        return f"$\\boxed{{{inside_box}}}$"
    else:
        return f"${inside_box}$"

SUBSTITUTIONS = [
    ('an ', ''), ('a ', ''), ('.$', '$'), ('\\$', ''), (r'\ ', ''),
    (' ', ''), ('mbox', 'text'), (',\\text{and}', ','), ('\\text{and}', ','),
    ('\\text{m}', '\\text{}'), ('cfrac', 'frac'), ('x\\in', '')
]

REMOVED_EXPRESSIONS = [
    'square', 'ways', 'integers', 'dollars', 'mph', 'inches', 'ft',
    'hours', 'km', 'units', '\\ldots', 'sue', 'points', 'feet',
    'minutes', 'digits', 'cents', 'degrees', 'cm', 'gm', 'pounds',
    'meters', 'meals', 'edges', 'students', 'childrentickets', 'multiples',
    '\\text{s}', '\\text{.}', '\\text{\ns}', '\\text{}^2',
    '\\text{}^3', '\\text{\n}', '\\text{}', r'\mathrm{th}', r'^\circ',
    r'^{\circ}', r'\;', r',\!', '{,}', '"', '\\dots', r'\\text{}', r'x\\in'
]

def normalize_final_answer(final_answer: str) -> str:
    """
    Remove extraneous dollar signs and noise, then perform simple text substitutions.
    """
    stripped = final_answer.strip('$').replace(',', '').strip()
    if re.match(r'^-?\d+$', stripped):
        return stripped
    final_answer = final_answer.replace('$$', '$').replace('\\(', '').replace('\\)', '')
    boxed_match = re.search(r'\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}', final_answer)
    if boxed_match:
        final_answer = boxed_match.group(1)
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, '')
    return final_answer.strip('$ ').strip()

def timeout_handler(signum, frame):
    raise TimeoutError("Symbolic comparison timed out")

def check_symbolic_equivalence(pred: str, target: str, timeout: int = 5) -> bool:
    """
    Compare two answers symbolically using sympy.
    For expressions containing commas or parentheses (e.g. intervals), fallback to direct string comparison.
    """
    if ',' in pred or pred.startswith('(') or pred.startswith('['):
        return pred == target
    if not pred or not target:
        return False
    try:
        if re.match(r'^-?\d+$', pred) and re.match(r'^-?\d+$', target):
            return int(pred) == int(target)
    except Exception:
        return False
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
        pred_sympy = parse_latex(r'' + pred)
        target_sympy = parse_latex(r'' + target)
        result = sympy.simplify(pred_sympy - target_sympy) == 0
        signal.alarm(0)
        return result
    except TimeoutError:
        print("Symbolic comparison timed out")
        return False
    except Exception as e:
        print(f"Error in comparison: {e}")
        return False

def evaluate_answer(model_response: str, ground_truth: str) -> Tuple[str, str, bool]:
    """
    Extract the final answer from a model response and the ground truth, normalize both,
    and then compare symbolically.
    Returns (normalized_pred, normalized_truth, is_correct).
    """
    pred_answer = extract_final_answer(model_response)
    truth_answer = extract_final_answer(ground_truth)
    norm_pred = normalize_final_answer(pred_answer)
    norm_truth = normalize_final_answer(truth_answer)
    is_correct = check_symbolic_equivalence(norm_pred, norm_truth)
    return norm_pred, norm_truth, is_correct

# ---------------------------
# 5. Evaluate accuracy for each condition
# ---------------------------
for key in TO_ANALYZE:
    TO_ANALYZE[key]["correct_or_not"] = False
    for i in range(len(TO_ANALYZE[key])):
        a = TO_ANALYZE[key]["first_turn_response"].iloc[i]
        b = TO_ANALYZE[key]["ground_truth_answer"].iloc[i]
        _, _, is_correct = evaluate_answer(a, b)
        TO_ANALYZE[key].at[i, "correct_or_not"] = is_correct
    accuracy = TO_ANALYZE[key]["correct_or_not"].sum() / len(TO_ANALYZE[key])
    print(f"Accuracy for {key}: {accuracy:.3f}")

incorrect_indices = TO_ANALYZE["cot"][TO_ANALYZE["cot"]["correct_or_not"]==False]["problem_idx"]
print("Problem indices with incorrect answers (COT):")
print(incorrect_indices.tolist())


print("\nSample results:")
print(tabulate(TO_ANALYZE["cot"].head(10), headers='keys', tablefmt='psql'))
