import re

PROMPT_TEMPLATE = """{{#system}}
You are a renowned mathematician known for your flawless accuracy and clarity. You solve math problems step by step,
using well-structured logic.
Always follow this exact response format:
1. Put your step-by-step calculation process inside <think> tags, explaining each step clearly.
2. Provide the final answer in a <boxed> tag, using a clear and simplified format.

Below are two examples. You must never deviate from this format.
Example 1:
{{#user}}
Lucy has 18 apples. She gives 4 apples to her friend. She then doubles the number of apples she has. How many apples does Lucy have left?
{{#assistant}}
<think>
1. Subtract the apples Lucy gave away: 18 - 4 = 14
2. Double the remaining apples: 14 * 2 = 28
</think>
\\boxed{28}

Example 2:
{{#user}}
What is the value of (3 + 5) * 2?
{{#assistant}}
<think>
1. Calculate the expression inside parentheses: 3 + 5 = 8
2. Multiply the result by 2: 8 × 2 = 16
</think>
\\boxed{16}

{{#user}}
$question
{{#assistant}}
"""

# TODO: Load this from config
max_length = 512

def make_preprocessor(tokenizer, max_length):

    def preprocess(batch):
        
        prompts = [PROMPT_TEMPLATE.replace("$question", q) for q in batch["question"]]
        inputs  = tokenizer(prompts, truncation=False, padding=False)

        # extract the number after "####"
        nums = []
        for ans in batch["answer"]:
            m = re.search(r"####\s*([-+]?\d*\.?\d+)", ans)
            if not m:
                raise ValueError(f"couldn't parse answer {ans!r}")
            nums.append(float(m.group(1)))
        inputs["labels"] = nums
        return inputs

    return preprocess