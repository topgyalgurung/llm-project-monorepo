from anthropic import Anthropic 

client = Anthropic()

model = "claude-sonnet-4-0"

# helper functions 
def add_user_message(messages, text):
	user_message = {"role": "user", "content": text}
	messages.append(user_message)

def add_assistant_message(messages, text):
	assistant_message = {"role": "user", "content": text}
	messages.append(assistant_message)


# basic streaming implementation 
messages = []
add_user_message(messages, "write 1 sentence of a fake dataset")
stream = client.messages.create(
	model=model,
	max_tokens=1000,
	messages=messages,
	stream=True
)

for event in stream:
	print(event)


# simplified text streaming 

with client.messages.stream(
	model=model,
	max_tokens=1000,
	messages=messages
) as stream:
	for text in stream.text_stream:
		# sent each chunk to client
		print(text, end="")
	# Get the complete message for database storage
    final_message = stream.get_final_message()

## Streaming exercises
# Use message prefilling and stop sequences 

def chat(messages, system=None, temperature=1.0):
    params = {
        "model": model,
        "max_tokens": 1000,
        "messages": messages,
        "temperature": temperature
    }
    if system:
        params["system"] = system

    message = client.messages.create(**params)
    return message.content[0].text

messages=[]

prompt="""
Generate three different sample AWS CLI commands. Each should be very short.
"""

add_user_message(messages, prompt)
add_assistant_message(messages, "Here are all three commands in a single block without any comments:\n```bash")

text = chat(messages, stop_sequences=["```"])
text.strip()

