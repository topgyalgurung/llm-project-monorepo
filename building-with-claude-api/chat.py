from dotenv import load_dotenv
from anthropic import Anthropic
import os

load_dotenv()

claude_api_key=os.getenv("ANTHROPIC_API_KEY")

if not claude_api_key:
      raise RuntimeError("ANTHROPIC_API_KEY is not set")

client = Anthropic(api_key=claude_api_key)
model = "claude-sonnet-4-0"

def chat(messages):
    message = client.messages.create(
        model=model,
        max_tokens = 1000,
        messages = messages,
    )
    return message.content[0].text

# helper functions 
def add_user_message(messages, text):
	user_message = {"role": "user", "content": text}
	messages.append(user_message)

def add_assistant_message(messages, text):
	assistant_message = {"role": "user", "content": text}
	messages.append(assistant_message)

messages = []

while True:
    user_input = input("> ")
    print(">", user_input)

    # add user input to the list of messages 
    add_user_message(messages, user_input)

    # call claude with chat function 
    answer = chat(messages)

    # add generated text to the list of messages 
    add_assistant_message(messages,answer)

    # print the generated text 
    print(answer)