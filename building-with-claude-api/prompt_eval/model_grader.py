
from prompt_eval.generate_test_dataset import add_assistant_message
from prompt_eval.generate_test_dataset import add_user_message
from prompt_eval.generate_test_dataset import chat
from prompt_eval.generate_test_dataset import generate_dataset
from statistics import mean

import json
import sys
from pathlib import Path

dataset_file = Path("dataset.json")

# Function to grade a test case + output using a model
def grade_by_model(test_case, output):
    eval_prompt = f"""
You are an expert AWS code reviewer. Your task is to evaluate the following AI-generated solution.

Original Task:
<task>
{test_case["task"]}
</task>

Solution to Evaluate:
<solution>
{output}
</solution>

Output Format
Provide your evaluation as a structured JSON object with the following fields, in this specific order:
- "strengths": An array of 1-3 key strengths
- "weaknesses": An array of 1-3 key areas for improvement
- "reasoning": A concise explanation of your overall assessment
- "score": A number between 1-10

Respond with JSON. Keep your response concise and direct.
Example response shape:
{{
    "strengths": string[],
    "weaknesses": string[],
    "reasoning": string,
    "score": number
}}
    """

    messages = []
    add_user_message(messages, eval_prompt)
    add_assistant_message(messages, "```json")
    eval_text = chat(messages, stop_sequences=["```"])
    return json.loads(eval_text)


test_case = {
    "task": "Create a Python function to extract an AWS account ID from an ARN"
}

# Passes a test case into Claude
def run_prompt(test_case):
    prompt = f"""
Please solve the following task:

{test_case["task"]}
"""

    messages = []
    add_user_message(messages, prompt)
    output = chat(messages)
    return output

# Function to execute a single test case and grade the output
def run_test_case(test_case):
    """Calls run_prompt, then grades the result"""
    output = run_prompt(test_case)

    model_grade = grade_by_model(test_case, output)
    score = model_grade["score"]
    reasoning = model_grade["reasoning"]

    return {
        "output": output,
        "test_case": test_case,
        "score": score,
        "reasoning": reasoning,
    }

# coordinate entire evaluation process
def run_eval(dataset):
    """Loads the dataset and calls run_test_case with each case"""
    results = []

    for test_case in dataset:
        result = run_test_case(test_case)
        results.append(result)

    average_score = mean([result["score"] for result in results])
    print(f"Average score: {average_score}")

    return results


def main():

    if dataset_file.exists():
        with dataset_file.open("r") as f:
            dataset = json.load(f)
    else:
        dataset = generate_dataset()

        with dataset_file.open("w") as f:
            json.dump(dataset,f,indent=2)
    results = run_eval(dataset)

    print(json.dumps(results, indent=2))


if __name__== "__main__":
    sys.exit(main())