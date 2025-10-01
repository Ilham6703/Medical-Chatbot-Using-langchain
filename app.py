from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI  # <<< CHANGED: Switched to Gemini
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os


app = Flask(__name__)


load_dotenv()

# --- Environment Variable Retrieval ---
# We retrieve the keys safely using os.getenv()
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')  # <<< RETRIEVING GOOGLE KEY

# NOTE: The redundant lines like os.environ["KEY"] = KEY have been removed here
# and in store_index.py to prevent the TypeError we debugged.

if not PINECONE_API_KEY:
    print("ERROR: PINECONE_API_KEY is not set.")
if not GOOGLE_API_KEY:
    # This check ensures the app fails early if the crucial key is missing
    print("ERROR: GOOGLE_API_KEY is not set. The LLM will not function.")


# --- Pinecone Setup (Uses free HuggingFace embeddings) ---
embeddings = download_hugging_face_embeddings()
index_name = "medical-chatbot" 

# Connect to the existing Pinecone index created by store_index.py
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

# --- LLM Initialization (Switched to Gemini) ---
# Using gemini-2.5-flash: a high-quality, free-tier-friendly model.
# LangChain automatically uses the GOOGLE_API_KEY environment variable.
chatModel = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.0  # Keep temperature low for factual RAG responses
)

# --- RAG Chain Setup ---
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(f"User Input: {input}")
    
    # Check for API key before invoking to prevent uncaught exceptions
    if not GOOGLE_API_KEY:
        error_msg = "AI is disabled. Missing GOOGLE_API_KEY."
        print(error_msg)
        return jsonify({"error": error_msg}), 500

    try:
        # Pass the input to the RAG chain
        response = rag_chain.invoke({"input": msg})
        
        answer = response.get("answer", "Error: Could not retrieve answer from Gemini.")
        print("Response : ", answer)
        return str(answer)

    except Exception as e:
        error_message = f"An error occurred while invoking the RAG chain: {e}"
        print(f"FATAL RAG ERROR: {error_message}")
        # Return a 500 status on failure for client handling
        return jsonify({"error": f"Failed to get response from AI: {str(e)}."}), 500


if __name__ == '__main__':
    # Flask development server for local testing only. Render ignores this block.
    app.run(host="0.0.0.0", port=8080, debug=True)
