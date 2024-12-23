from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

def main(query):
    # Let the user know we're working on their response
    print("\nPlease wait, your response is being generated!\n")

    # Initialize the language model you want to use.
    # Here, we're using a model called "llama2" provided by the Ollama API.
    llm = Ollama(model="llama2")

    # Define the structure of the prompt to guide the model's response.
    # The prompt tells the model to think step by step and answer based only on context.
    prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context. Think step by step before providing a detailed answer. 
    <context>
    {context}
    </context>
    Question: {input}""")

    # Specify where your Chroma database is stored. This is used for efficient retrieval.
    persist_directory = "chroma_db"

    # Create an embedding model for vectorizing text. Here, we use a lightweight HuggingFace model.
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Initialize the Chroma vector store with your embeddings and database path.
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    # Combine the language model and the prompt into a chain to process documents.
    document_chain = create_stuff_documents_chain(llm, prompt)

    # Turn your Chroma database into a retriever to fetch relevant context for a query.
    retriever = db.as_retriever()

    # Create the full retrieval-augmented generation (RAG) pipeline.
    # This connects the retriever and the document chain for a seamless process.
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # Use the chain to process the input query and generate a response.
    response = retrieval_chain.invoke({"input": query})

    # Print the final response to the user.
    print(response['answer'])

if __name__ == "__main__":
    
    # Ask the user for their query and pass it to the main function.
    query = str(input("Enter the Query - "))
    main(query)