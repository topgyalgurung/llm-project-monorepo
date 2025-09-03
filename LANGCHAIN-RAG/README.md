# Langchain RAG 

Build a RAG (Retrieval Augmented Generation) app with Langchain and OpenAI in Python to create chat-bots for documents, books or files.

- Database: Chroma 
- Langchain
- Open AI API

### Steps:
- Prepare the data:
  - load documents, split text to get chunks and save to Chroma db
- then query the database by taking a query then turn that into embeddings using same function
- scan through db and find e.g 5 chunks info closest to embedding distance from our query
- then we can put that together and have AI read that info and decide response to user 
- creating response:
  - feed relevant data into open ai to create high quality response using that data as source. we can create PROMPT_TEMPLATE

#### Embeddings:

- Vector Embeddings: 
  - Embeddings are vector representation of text that capture their meaning. In python is a list of numbers
- distance between these vectors can be calculated using cosine similarity or euclidean distance
- to generate a vector from a word, we will need an LLM like Open AI api (usually api or function we can call)
- langchain gives utility function to compare embedding distance directly using Open AI 


## Install dependencies

- for mac 
```bash
 conda install onnxruntime -c conda-forge
 ```

- for windows: see this [thread](https://github.com/microsoft/onnxruntime/issues/11037)

- Setup local environment 
```bash
    python3 -m venv venv 
    source venv/bin/activate 
    pip freeze --local > requirements.txt
    pip install -r requirements.txt 
```
- check requirements.txt file for required dependencies

## Create database 
Create the Chroma DB
```bash
python create_datbase.py
```

## Query the database 
Query the Chroma DB.
```bash
python query_data.py "How does Alice meet the Mad Hatter"
```

Make sure to set up an Open AI Account, set API key in .env 