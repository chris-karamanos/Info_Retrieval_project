from langchain_ollama import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# Initialize the Chat Model
def initialize_chat_model():
    return ChatOllama(model="phi", temperature=0.5)

# Create a chatbot for a specific set of documents
def create_chatbot(documents, llm):
    # Split documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = [text_splitter.create_documents([doc]) for doc in documents]
    flat_chunks = [chunk for sublist in chunks for chunk in sublist]

    # Use SentenceTransformers for embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(flat_chunks, embeddings)

    # Set up the retriever
    retriever = vector_store.as_retriever()

    # Create the RetrievalQA chain
    qa_pipeline = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True  # This will return source docs along with the answer
    )

    return qa_pipeline

# Function to load and parse the documents that our model returned 
def load_documents(file_path):
    """Load documents from a file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return [doc.strip() for doc in content.split("\n\n") if doc.strip()]


# Initialize the chat model
llm = initialize_chat_model()


# Command-line interface to interact with the chatbot
def chat_with_bot(chatbot_name, chatbot):
    print(f"\n{chatbot_name} Chatbot is ready! Type 'exit' to quit.\n")
    while True:
        query = input("Your question: ")
        if query.lower() == "exit":
            print("Goodbye!")
            break
        try:
            response = chatbot.invoke({"query": query})
            # Extract and display the result
            print(f"{chatbot_name} Response: {response['result']}")
        except Exception as e:
            print(f"Error: {e}")

# Run the two chatbots
if __name__ == "__main__":

    # Load documents from the file generated by the Boolean and VSM model
    bool_documents = load_documents("../Task2/bool_retrieved_docs.txt")
    vsm_folder_path = "../Task3/vsm_retrieved_docs"


    print("Select Chatbot: 1 for Boolean, 2 for VSM")
    choice = input("Enter your choice: ")
    if choice == "1":
        boolean_chatbot = create_chatbot(bool_documents, llm)
        chat_with_bot("Boolean", boolean_chatbot)
    elif choice == "2":
        # Ask the user to select a query ID
        print("Choose a query (1-20) for the VSM chatbot:")
        query_choice = input("Enter query number: ")
        
        # Construct the path to the selected VSM file
        vsm_file_path = f"{vsm_folder_path}/vsm_retrieved_docs_query_{query_choice}.txt"

        try:
            # Load the selected VSM document file
            vsm_documents = load_documents(vsm_file_path)
            
            # Create and test the VSM chatbot
            vsm_chatbot = create_chatbot(vsm_documents, llm)
            chat_with_bot("VSM", vsm_chatbot)

        
        except FileNotFoundError:
            print(f"Error: File for query {query_choice} not found in {vsm_folder_path}.")
    else:
        print("Invalid choice!") 


