from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
import openai
from dotenv import load_dotenv
import os
import shutil 
 
load_dotenv()

openai.api_key = os.environ['OPENAI_API_KEY']

DATA_PATH = "data/books"
CHROMA_PATH = "chroma"

def main():
    generate_data_store()

def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    # save to db

def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob = "*.md")
    documents = loader.load()
    return documents 

# to be more focused and relevant to what we looking for use recursovecharactertextsplitter
def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 300, # of chars
        chunk_overlap = 100, # each chunk overlap of # chars
        length_function = len,
        add_start_index = true,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    # to see sample chunks
    document = chunks[10]
    print(document.page_content)
    print(document.metadata)

def save_to_chroma(chunks:list[Document]):
    # clear out prev version of database first
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    
    # create a new db from the documents
    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
    )

    # database should save auto but we can force to save 
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}")

if __name__ == "__main__":
    main()