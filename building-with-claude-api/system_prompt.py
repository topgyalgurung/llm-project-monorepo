def chat(messages, system=None):
    params ={
        "model"=model,
        "max_tokens" = 1000,
        "messages" = messages
    }
    if system:
        params["system"] = system
    message = client.messages.create(**params)
    return message.content[0].text


messages = []


# without system prompt
answer = chat(messages)

# with system prompt
system = """
you are a patient math tutor. do not directly answer student's question.
Guide them to a solution step by step
"""

add_user_message(messages, system)