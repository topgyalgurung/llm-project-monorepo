
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

# helper functions 
def add_user_message(messages, text):
	user_message = {"role": "user", "content": text}
	messages.append(user_message)

     
messages = []

add_user_message(
    messages,
    "Generate a one sentence movie idea"
)
answer = chat(messages, temperature=1.0)
answer