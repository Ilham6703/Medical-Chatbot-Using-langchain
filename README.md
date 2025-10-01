🩺 Medibot – AI-Powered Medical Chatbot

Medibot is a state-of-the-art medical chatbot built using a Retrieval-Augmented Generation (RAG) system. Leveraging the LangChain framework, it delivers accurate, context-aware medical responses by combining a custom knowledge base with powerful LLMs. The web interface is served via Flask, making it fast, accessible, and production-ready.

🚀 Features

Context-Aware Answers: Medibot answers medical questions using only authoritative content from your PDF knowledge base.

High-Performance Embeddings: Uses HuggingFace MiniLM-L6-v2 embeddings for semantic search.

Vector Database: Pinecone stores and retrieves relevant text chunks instantly.

Reliable LLM: Powered by ChatGoogleGenerativeAI (Gemini-2.5-flash) for accurate, human-like responses.

Easy Deployment: Web app built with Flask and deployable with Gunicorn.

Cost-Effective: Free tools combined with advanced vector search ensure performance without breaking the bank.

🏗 Architecture Overview

| Component               | Implementation                                            | Purpose                                                                          |
| ----------------------- | --------------------------------------------------------- | -------------------------------------------------------------------------------- |
| **Data Source**         | PDF files in `data/` directory                            | Custom medical knowledge base                                                    |
| **Document Processing** | `src/helper.py` using `PyPDFLoader` & `DirectoryLoader`   | Loads PDFs, splits text into chunks, prepares for embedding                      |
| **Embeddings**          | `HuggingFaceEmbeddings` (MiniLM-L6-v2)                    | Converts text chunks into 384-dimensional vectors for semantic search            |
| **Vector Database**     | `PineconeVectorStore`                                     | Stores embeddings and retrieves semantically relevant chunks quickly             |
| **LLM (Generator)**     | `ChatGoogleGenerativeAI` (Gemini-2.5-flash)               | Generates answers based on retrieved context                                     |
| **Orchestration**       | `create_retrieval_chain` / `create_stuff_documents_chain` | Connects Retriever and LLM using a specialized Prompt Template for RAG           |
| **Web Server**          | Flask (`app.py`)                                          | Handles user queries and serves the web frontend (`/`) and API endpoint (`/get`) |

⚡ How It Works

Load PDFs: The bot reads PDFs from the data/ folder.

Process Documents: Text is split into manageable chunks for embedding.

Embed Text: Each chunk is converted into numerical vectors using HuggingFace embeddings.

Store in Pinecone: Vectors are indexed in Pinecone for fast semantic retrieval.

Retrieve & Generate: When a user asks a question, the most relevant chunks are retrieved and fed into the LLM (Gemini) to generate a context-aware answer.

Serve via Web: Flask handles queries and displays responses in a user-friendly interface.