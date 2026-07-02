from urllib import response

from dotenv import load_dotenv
from anthropic import Anthropic
import os 

#load environment variable 
load_dotenv()

my_api_key = os.getenv("ANTHROPIC_API_KEY")
if not my_api_key:
    raise RuntimeError("ANTHROPIC_API_KEY is not set in environment")

client = Anthropic(api_key=my_api_key)

conversation_history = []

while True:
    user_input = input("User: ")
    if user_input.lower() == "quit":
        print("conversation ended.")
        break
    conversation_history.append({"role": "user", "content": user_input})

    response = client.messages.create(
        model = "claude-haiku-4-5-20251001",
        messages = conversation_history,
        max_tokens = 500
    )

    assistant_response = response.content[0].text
    print(f"Assistant: {assistant_response}")
    conversation_history.append({"role": "assistant", "content": assistant_response})
