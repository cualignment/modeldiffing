import re

def check_format(generation):
    """
    Check if the generation follows the required format:
    1. Contains <think> tags with step-by-step calculations.
    2. Contains a \boxed tag with the final answer.
    """
    if "<think>" not in generation or "</think>" not in generation:
        return False
    if "\\boxed" not in generation:
        return False
    return True

def get_numerical_answer(generation):
    """
    Extract the numerical answer from the generation.
    Either from the \boxed{} tag or the last number in the generation.
    """
    # Try to find the boxed answer
    boxed_match = re.search(r"\\boxed{([-+]?\d*\.?\d+)}", generation)
    if boxed_match:
        return float(boxed_match.group(1))
    
    # If not found, try to find the last number in the generation
    last_number_match = re.search(r"([-+]?\d*\.?\d+)(?!.*[-+]?\d*\.?\d+)", generation)
    if last_number_match:
        return float(last_number_match.group(1))
    
    return None

def check_answer(generation, answer, eps=1e-3):
    """
    Computes the reward for each generation answer pair.
    First we need to split the generation at {#assistant} and take the third section to remove the prompt.
    The reward is 0.5 for the correct formatting and another 0.5 for the correct answer (generation and answer match up to eps difference)
    """
    reward = [None for _ in range(len(generation))]
    for idx, (g, a) in enumerate(zip(generation, answer)):
        r = 0.0
        if check_format(g):
            r += 0.25
        numerical_answer = get_numerical_answer(g)
        print(f"numerical answer: {numerical_answer}, expected: {a}, eps: {eps}")
        if numerical_answer is not None and abs(numerical_answer - a) < eps:
            r += 0.75
        reward[idx] = r
    return reward