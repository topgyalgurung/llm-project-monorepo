
stream = client.messages.create(
    messages=[
        {
            "role": "user",
            "content": "Write me a 3 word sentence, without a preamble.  Just give me 3 words",
        }
    ],
    model="claude-3-haiku-20240307",
    max_tokens=100,
    temperature=0,
    stream=True,
)

for event in stream:
    if event.type == "message_start":
        input_tokens = event.messsage.usage.input_tokens
        print("MESSAGE START EVENT", flush=True)
        print(f"Input tokens used: {input_tokens}", flush=True)
    elif event.type == "content-block-delta":
        print(event.delta.text, flush=True, end="")
    elif event.type == "message_delta":
        output_tokens = event.usage.output_tokens
        print("\n========================", flush=True)
        print("MESSAGE DELTA EVENT", flush=True)
        print(f"Output tokens used: {output_tokens}", flush=True)



import time
def measure_non_streaming_ttft():
    start_time = time.time()

    response = client.messages.create(
        max_tokens=500,
        messages=[
            {
                "role": "user",
                "content": "Write mme a long essay explaining the history of the American Revolution",
            }
        ],
        temperature=0,
        model="claude-3-haiku-20240307",
    )

    response_time = time.time() - start_time

    print(f"Time to receive first token: {response_time:.3f} seconds")
    print(f"Time to recieve complete response: {response_time:.3f} seconds")
    print(f"Total tokens generated: {response.usage.output_tokens}")
    
    print(response.content[0].text)


from anthropic import AsyncAnthropic

client = AsyncAnthropic()

async def streaming_with_helpers():
    async with client.messages.stream(
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": "Write me sonnet about orchids",
            }
        ],
        model="claude-3-opus-20240229",
    ) as stream:
        async for text in stream.text_stream:
            print(text, end="", flush=True)
    final_message = await stream.get_final_message()
    print("\n\nSTREAMING IS DONE.  HERE IS THE FINAL ACCUMULATED MESSAGE: ")
    print(final_message.to_json())
await streaming_with_helpers()


# MyStream class  with two event handler

class MyStream(AsyncMessageStream):
    async def on_text(self, text, snapshot):
        # This runs only on text delta stream messages
        print(green + text + reset, flush=True) #model generated content is printed in green

    async def on_stream_event(self, event):
        # This runs on any stream event
        print("on_event fired:", event.type)