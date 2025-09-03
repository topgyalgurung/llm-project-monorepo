from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


DATA_PATH = "data/books"

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
def split_text(documents: list[Documents]):
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


if __name__ == "__main__":
    main()