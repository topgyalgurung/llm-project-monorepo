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

def translate_word(word, language):
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1000,
        messages=[
            {
                "role": "user",
                "content": f"Translate the word {word} into {language}. Only respond with the translated word, nothing else"
            }
        ]
    )
    return response.content[0].text

def few_shot_prompt():
    response = client.messages.create(
    model="claude-3-haiku-20240307",
    max_tokens=500,
    messages=[
        {"role": "user", "content": "Unpopular opinion: Pickles are disgusting. Don't @ me"},
        {"role": "assistant", "content": "NEGATIVE"},
        {"role": "user", "content": "I think my love for pickles might be getting out of hand. I just bought a pickle-shaped pool float"},
        {"role": "assistant", "content": "POSITIVE"},
        {"role": "user", "content": "Seriously why would anyone ever eat a pickle?  Those things are nasty!"},
        {"role": "assistant", "content": "NEGATIVE"},
        {"role": "user", "content": "Just tried the new spicy pickles from @PickleCo, and my taste buds are doing a happy dance! 🌶️🥒 #pickleslove #spicyfood"},
        ]
    )
    print(response.content[0].text)

if __name__ == "__main__":
    # Example usage
    print(translate_word("hello", "Spanish"))  # Should output "hola"