# build agent from scratch

import openai
import re
import httpx
import os
from dotenv import load_dotenv

_ = load_dotenv

from openai import OpenAI

client = OpenAI()

# chat_completion = client.chat.completions.create(
#     model = "gpt-3.5-turbo",
#     messages = [{"role", "content": "hello world "}]
# )

# chat_completion.choices[0].messages.content

class Agent:
    def __init__(self, system=""):
        self.system = system
        self.messages = []
        if self.system:
            self.messages.append({"role": "system", "content":system })

    def __call__(self, message):
        self.messages.append({"role": "user", "content": message})
        result = self.execute()

    def execute(self):
        completion = client.chat.completions.create(
                        model = "gpt-4o",
                        temperature=0,
                        messages=self.messages)
        return completion.choices[0].message.content

