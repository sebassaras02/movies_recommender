import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings
import os

pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV"))

# FUNCTION TO CREATE A RETRIVER
def create_retriever():
    """
    This function creates a Pinecone vector store object with the OpenAI embeddings.

    Args:
        None

    Returns:
        vector_store: Pinecone vector store object with the OpenAI embeddings.
    """
    pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV"))

    # Load the embeddings
    embedding = OpenAIEmbeddings(api_key = os.getenv("OPENAI_API_KEY"))

    # Create a Pinecone vector store
    vector_store = Pinecone.from_existing_index(index_name="movies", embedding=embedding, namespace="platzi")

    return vector_store.as_retriever(search_kwargs={"k": 5})


# FUNCTION TO SEARCH FOR SIMILAR BOOKS
def search_similar_books(retriever, input, metadata=None, use_metadata=False):
    """
    This function searches for similar books given an input text.

    Args:
        retriever: Pinecone vector store object.
        input (str): input text.
        metadata (str): metadata to filter the search.
    Returns:
        

    """
    if metadata:
        results = retriever.get_relevant_documents(input, metadata={"categories":metadata})
    else:
        results = retriever.get_relevant_documents(input)
    
    return results