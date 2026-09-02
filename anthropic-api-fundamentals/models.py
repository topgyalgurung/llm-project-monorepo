from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

client = Anthropic()

import time
def compare_model_speeds():
    models = ["claude-3-5-sonnet-20240620","claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"]
    task = "Explain the concept of photosynthesis in a concise paragraph."

    for model in models:
        start_time = time.time()

        response = client.messages.create(
            model=model,
            max_tokens=500,
            messages=[{"role": "user", "content": task}]
        )

        end_time = time.time()
        execution_time = end_time - start_time
        tokens = response.usage.output_tokens
        time_per_token = execution_time/tokens

        print(f"Model: {model}")
        print(f"Response: {response.content[0].text}")
        print(f"Generated Tokens: {tokens}")
        print(f"Execution Time: {execution_time:.2f} seconds")
        print(f"Time Per Token: {time_per_token:.2f} seconds\n")

compare_model_speeds()


def compare_model_capabilities():
    models = ["claude-3-5-sonnet-20240620", "claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"]
    task = """
    What is the geometric monthly fecal coliform mean of a distribution system with the following FC
 counts: 24, 15, 7, 16, 31 and 23? The result will be inputted into a NPDES DMR, therefore, round
 to the nearest whole number.  Respond only with a number and nothing else.
    """

    for model in models:
        answers = []
        for attempt in range(7):
            response = client.messages.create(
                model=model,
                max_tokens=1000,
                messages=[{"role": "user", "content": task}]
            )
            answers.append(response.content[0].text)

        print(f"Model: {model}")
        print(f"Answers: ", answers)

compare_model_capabilities()