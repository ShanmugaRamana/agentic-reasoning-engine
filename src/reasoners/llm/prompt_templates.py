# src/reasoners/llm/prompt_templates.py

PROMPT_TEMPLATES = {
    "base": """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a precise logical reasoner. Your task is to solve the problem by following these steps:
1.  First, think step-by-step inside the <scratchpad> tags.
2.  Then, provide your final answer inside the <answer> tags. Your final answer must be ONLY the single number of the correct option.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Problem: "{problem_statement}"

Options:
1: {answer_option_1}
2: {answer_option_2}
3: {answer_option_3}
4: {answer_option_4}
5: {answer_option_5}

Follow the required format for your response.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
<scratchpad>
(My reasoning process starts here...)
</scratchpad>
<answer>
(My final answer, a single digit, goes here)
</answer>
""",

    "classic riddles": """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a master riddle solver.
1. First, think step-by-step about the wordplay and logical trap inside the <scratchpad> tags.
2. Then, provide your final answer inside the <answer> tags. Your final answer must be ONLY the single number of the correct option.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Example Problem: "You are in a race and you overtake the second person. What position are you in now?"
Options: 1: First, 2: Second, 3: Third, 4: Fourth, 5: Another answer
Correct Response:
<scratchpad>
The riddle is a logical trap. If I overtake the person who is in second place, I take their position. I am now in second place. The person I overtook is now in third.
</scratchpad>
<answer>
2
</answer>

---
New Problem: "{problem_statement}"

Options:
1: {answer_option_1}
2: {answer_option_2}
3: {answer_option_3}
4: {answer_option_4}
5: {answer_option_5}

Follow the required format for your response.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
""",
    
    "spatial reasoning": """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert in spatial and geometric reasoning.
1. First, think step-by-step inside the <scratchpad> tags, visualizing the problem.
2. Then, provide your final answer inside the <answer> tags. Your final answer must be ONLY the single number of the correct option.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Problem: "{problem_statement}"

Options:
1: {answer_option_1}
2: {answer_option_2}
3: {answer_option_3}
4: {answer_option_4}
5: {answer_option_5}

Follow the required format for your response.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
<scratchpad>
1. Visualize the setup described in the problem.
2. Identify all spatial constraints and relationships.
3. Perform any necessary calculations or logical deductions.
4. Compare the result with the given options.
</scratchpad>
<answer>
(My final answer, a single digit, goes here)
</answer>
""",

    "lateral thinking": """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert in lateral thinking and finding non-obvious solutions.
1. First, analyze the problem for double meanings and hidden assumptions inside the <scratchpad> tags.
2. Then, provide your final answer inside the <answer> tags. Your final answer must be ONLY the single number of the correct option.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Example Problem: "A woman shoots her husband, holds him underwater, and hangs him. 5 minutes later, they enjoy dinner together. How?"
Options: 1: Twin sister, 2: Photography process, 3: He's a ghost, 4: They made up, 5: Another answer
Correct Response:
<scratchpad>
This is a lateral thinking puzzle where words have double meanings. "Shoots" refers to taking a picture. "Holds him underwater" refers to developing the film. "Hangs him" refers to hanging the photo to dry.
</scratchpad>
<answer>
2
</answer>

---
New Problem: "{problem_statement}"

Options:
1: {answer_option_1}
2: {answer_option_2}
3: {answer_option_3}
4: {answer_option_4}
5: {answer_option_5}

Follow the required format for your response.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
""",

    "operation of mechanisms": """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a systems engineer who analyzes processes and state changes.
1. First, model the system and simulate the actions step-by-step inside the <scratchpad> tags.
2. Then, provide your final answer inside the <answer> tags. Your final answer must be ONLY the single number of the correct option.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Problem: "{problem_statement}"

Options:
1: {answer_option_1}
2: {answer_option_2}
3: {answer_option_3}
4: {answer_option_4}
5: {answer_option_5}

Follow the required format for your response.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
<scratchpad>
1. Identify the components and their initial states.
2. List the allowed actions and their effects on the states.
3. Simulate the sequence of actions to determine the final outcome.
4. Compare the outcome to the options.
</scratchpad>
<answer>
(My final answer, a single digit, goes here)
</answer>
""",

    "sequence solving": """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a mathematician specializing in number sequences.
1. First, find the underlying pattern by analyzing the differences between terms inside the <scratchpad> tags.
2. Then, provide your final answer inside the <answer> tags. Your final answer must be ONLY the single number of the correct option.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Example Problem: "A sequence of numbers: 1, 5, 12, 22, 35, ?"
Options: 1: 42, 2: 49, 3: 51, 4: 56, 5: Another answer
Correct Response:
<scratchpad>
1. The sequence is 1, 5, 12, 22, 35.
2. The first differences are 4, 7, 10, 13.
3. The second differences are 3, 3, 3. This is a constant, so the pattern is quadratic.
4. The next first difference will be 13 + 3 = 16.
5. The next term in the sequence is 35 + 16 = 51.
6. Option 3 matches the answer 51.
</scratchpad>
<answer>
3
</answer>

---
New Problem: "{problem_statement}"

Options:
1: {answer_option_1}
2: {answer_option_2}
3: {answer_option_3}
4: {answer_option_4}
5: {answer_option_5}

Follow the required format for your response.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
""",

    "optimization of actions and planning": """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert in optimization and planning.
1. First, find the most efficient solution by considering all constraints and objectives inside the <scratchpad> tags.
2. Then, provide your final answer inside the <answer> tags. Your final answer must be ONLY the single number of the correct option.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Problem: "{problem_statement}"

Options:
1: {answer_option_1}
2: {answer_option_2}
3: {answer_option_3}
4. {answer_option_4}
5: {answer_option_5}

Follow the required format for your response.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
<scratchpad>
1. List all tasks, their costs/durations, and any constraints.
2. Evaluate the possible sequences or paths.
3. Identify the path that minimizes cost/time or maximizes value, according to the problem.
4. Compare the result to the given options.
</scratchpad>
<answer>
(My final answer, a single digit, goes here)
</answer>
""",
    
    "logical traps": """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert at identifying logical traps and contradictory premises.
1. First, analyze the problem statement carefully for impossible conditions inside the <scratchpad> tags.
2. Then, provide your final answer inside the <answer> tags. Your final answer must be ONLY the single number of the correct option.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Example Problem: "In a square room, a lamp must be equidistant from corners A, B, and C, and also twice as far from corner D as it is from A. Where should it be placed?"
Options: 1: Center of room, 2: Corner A, 3: Halfway, 4: Nowhere, because it's a logical trap, 5: Another answer
Correct Response:
<scratchpad>
The premise requires the point to be the center of the circumcircle of triangle ABC, which is the center of the square. Let the center be P. The distances PA, PB, PC, and PD are all equal. The second premise requires PD = 2 * PA. Since PA = PD, this means PA = 2 * PA, which is only possible if PA = 0, but the lamp isn't at a corner. The constraints are contradictory.
</scratchpad>
<answer>
4
</answer>

---
New Problem: "{problem_statement}"

Options:
1: {answer_option_1}
2: {answer_option_2}
3: {answer_option_3}
4: {answer_option_4}
5: {answer_option_5}

Follow the required format for your response.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
}