"""
For this exercise, we'd like you to use Claude to transcribe and summarize an Anthropic research paper. In the images folder, you'll find  research_paper folder that contains 5 screenshots of a research paper. To help you out, we've provided all 5 image URLs in a list:
"""
from anthropic import Anthropic
from dotenv import load_dotenv
from IPython.display import Image

import base64
import mimetypes

load_dotenv()
client = Anthropic()

research_paper_pages = [
    "./images/research_paper/page1.png",
    "./images/research_paper/page2.png",
    "./images/research_paper/page3.png",
    "./images/research_paper/page4.png",
    "./images/research_paper/page5.png"
    ]

def create_image_message(image_path):
    with open(image_path, "rb") as image_file:
        binary_data = image_file.read()
    base64_encoded_data = base64.b64encode(binary_data)
    base64_string = base64_encoded_data.decode('utf-8')
    mimetype, _ =mimetypes.guess_type(image_path)

    # create image block
    image_block = {
        "type": "image",
        "source":{
            "type": "base64",
            "media_type": mimetype,
            "data": base64_string
        }
    }
    return image_block


def transcribe_single_page(page_url):
    messages=[
        {
        "role": "user",
        "content":[
            create_image_message(page_url),
            {"type":"text", "text":"transcribe the text from this page of a research paper as accurately as possible."}
        ]
        }
    ]
    response = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=5000,
        messages=messages
    )
    return response.content[0].text

def summarize_paper(pages):
    complete_paper_text=""
    for page in pages:
        print("transcribing page ", page)
        transcribed_text = transcribe_single_page(page)
        print(transcribed_text[:200])
        complete_paper_text += transcribed_text
    response = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=5000,
        messages=[
            {
                "role": "user",
                "content":f"This is the transcribed contents of a research paper <paper>{complete_paper_text}<paper>. Please 
                summarize this paper for a non-research audience in at least 3 paragragphs"
            }
        ]
    )
    print(response.content[0].text)

summarize_paper(research_paper_pages)


