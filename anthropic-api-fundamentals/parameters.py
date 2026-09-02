

def generate_questions(topic, num_questions=3):
    response = client.messages.create(
        model = "claude-3-haiku-20240307",
        max_tokens=1000,
        system=f"You are an expert on {topic}. generate thought provoking questions about topic. "
        messages=[
            {"role": "user", "content": f"Generate {num_questions} questions about {topic} as a numbered list"}
        ],
        stop_sequences = [f"{num_questions+1}"]
    )
    print(response.content[0].text)

generate_questions(topic="free will", num_questions=3)