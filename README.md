# Medical Operations RAG Agent

A RAG (Retrieval-Augmented Generation) system built with FastAPI, Streamlit, and LangChain to provide answers based on clinical compliance manuals and oncology guidelines.

## Tech Stack
- **Frontend:** Streamlit
- **Backend:** FastAPI
- **LLM:** Groq (Llama 3.1)
- **Vector Store:** Pinecone
- **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`)

## Setup

1. **Clone the repository**
2. **Create a `.env` file** in the root directory:
   ```env
   GROQ_API_KEY=your_groq_key
   PINECONE_API_KEY=your_pinecone_key
   PINECONE_INDEX_NAME=your_index_name
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Ingest Data:**
   Place your PDF manuals in the `./data` folder and run:
   ```bash
   python ingest.py
   ```

## Running the Application

1. **Start the Backend (FastAPI):**
   ```bash
   uvicorn backend:app --reload
   ```
2. **Start the Frontend (Streamlit):**
   ```bash
   streamlit run app.py
   ```
