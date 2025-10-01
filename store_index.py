from dotenv import load_dotenv
import os
from src.helper import load_pdf_file, filter_to_minimal_docs, text_split, download_hugging_face_embeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec 
from langchain_pinecone import PineconeVectorStore

# Load all environment variables from the .env file into os.environ
load_dotenv()

# --- Retrieve the key safely for use in the script ---
# We no longer set os.environ["KEY"] = KEY, which caused the TypeError 
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
# We do not retrieve GOOGLE_API_KEY or OPENAI_API_KEY here, as they are not needed for indexing.

# Add a check to confirm the key loaded properly
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is not set. Please check your .env file!")


extracted_data=load_pdf_file(data='data/')
filter_data = filter_to_minimal_docs(extracted_data)
text_chunks=text_split(filter_data)

# NOTE: Using HuggingFace Embeddings (free tier)
embeddings = download_hugging_face_embeddings()

pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key=pinecone_api_key)


index_name = "medical-chatbot"  # change if desired

# Ensure the correct Pinecone index is created before proceeding
if not pc.has_index(index_name):
    print(f"Creating Pinecone index: {index_name}...")
    pc.create_index(
        name=index_name,
        dimension=384, # Must match the HuggingFace embedding model dimension
        metric="cosine",
        # Keep 'aws' region here, Render will deploy from a specific region, but Pinecone is external.
        spec=ServerlessSpec(cloud="aws", region="us-east-1"), 
    )

print(f"Connecting to Pinecone index: {index_name}...")
index = pc.Index(index_name)

# This upserts the documents and creates the vector store
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings, 
)

print("Indexing complete! Your chatbot data is ready.")
