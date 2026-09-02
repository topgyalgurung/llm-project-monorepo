from dotenv import load_dotenv
load_dotenv()
import json

from anthropic import Anthropic

client = Anthropic()

def add_user_message(messages, text):
    user_message = {"role": "user", "content": text}
    messages.append(user_message)

def add_assistant_message(messages, text):
    assistant_message = {"role": "assistant", "content": text}
    messages.append(assistant_message)

def chat(messages, system=None, stop_sequences=[]):
    params = {
        "model": "claude-haiku-4-5",
        "max_tokens": 1000,
        "messages": messages,
    }
    if system:
        params["system"] = system
    if stop_sequences:
        params["stop_sequences"] = stop_sequences
    
    response = client.messages.create(**params)
    return response.content[0].text

# data set generation function 

def generate_dataset():
    prompt = """
    Generate an evaluation dataset for a prompt evaluation. 
    The dataset will be used to evaluate prompts that generate Python, JSON, or Regex specifically for AWS-related tasks. 
    Generate an array of JSON objects, each representing task that requires Python, JSON, or a Regex to complete.
    Example output:
    ```json
    [
        {
        "task": "Create a python function to extract aws account id from an arn",
        },
        {
            "task": "Write a JSON policy document that allows read-only access to a specific S3 bucket",
        },
    
    ]
```

* Focus on tasks that can be solved by writing a single Python function, a single JSON object, or a single regex
* Focus on tasks that do not require writing much code

Please generate 3 objects.
"""

    messages = []
    add_user_message(messages, prompt)
    add_assistant_message(messages, "```json")
    text = chat(messages, stop_sequences=["```"])
    return json.loads(text)

# test the dataset generation 

dataset = generate_dataset()
print(dataset)

# saving the dataset
with open('dataset.json', 'w') as f:
    json.dump(dataset, f, indent=2)