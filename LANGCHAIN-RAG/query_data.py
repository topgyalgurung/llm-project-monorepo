import argparse

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.promps import ChatPromptTemplate

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on following context:
{context}
---
Answer the question based on the above context: {question}

"""
def main():
    # argparse module to create user friendly command line interfaces (CLI)
    parser = argparse.ArgumentParser("simple program to demonstrate query data")
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args() # parse command line args by user
    query_text = args.query_text

    # prepare the db
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory = CHROMA_PATH, embedding_function=embedding_function)

    # search the db
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.7: # or relevance score below certain threshold we define return early
        print(f"Unable to find matching results")
        return

    # now we can feed into open ai 
    # creates the prompt 
    context_text = "\n\n --- \n\n".join([doc.page_content for doc, _score in results])
    # print(context_text) # print content for each page
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context= context_text, question = query_text)
    print(prompt)

    # uses LLM to answer the question 
    model = ChatOpenAI()
    response_text = model.predict(prompt)

    sources = [doc.metadata.get("source", None) for doc, _score in result]
    formatted_response = f"Response: {response_text}\n Sources: {sources}"
    print(formatted_response)

if __name__=="__main__":
    main()