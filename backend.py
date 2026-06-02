import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import JsonOutputParser # <-- NEW IMPORT

load_dotenv()

app = FastAPI()

# SECURITY NOTE: Replace "*" with your actual Streamlit frontend URL later 
# so other websites cannot secretly use your backend API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://medical-rag-agent-f2onvfxjd4wewbyudgcpe2.streamlit.app/"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = PineconeVectorStore(
    index_name=os.environ.get("PINECONE_INDEX_NAME"), 
    embedding=embeddings
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

class RAGResponse(BaseModel):
    answer: str = Field(description="The clinical or operational answer to the question.")
    citation: str = Field(description="The exact document name and page number. If the answer is NOT in the text, strictly output: 'Error: Protocol not found'")

class QueryRequest(BaseModel):
    question: str

@app.post("/ask", response_model=RAGResponse)
async def ask_bot(request: QueryRequest):
    try:
        # 1. Retrieve context
        docs = retriever.invoke(request.question)
        
        # --- SYNTAX FIX APPLIED HERE ---
        formatted_docs = []
        for d in docs:
            clean_source = d.metadata['source'].replace('\\', '/')
            page_num = d.metadata.get('page', 'Unknown')
            formatted_docs.append(f"Source: {clean_source} (Page {page_num})\n{d.page_content}")

        context = "\n\n".join(formatted_docs)
        # -------------------------------

        # 2. Setup JSON Parser
        parser = JsonOutputParser(pydantic_object=RAGResponse)

        # 3. Prompt Engineering
        system_prompt = f"""You are a strict Clinical Guidelines Support Bot. 
        Answer the question using ONLY the provided context. 
        If the answer is not in the context, your citation MUST be 'Error: Protocol not found' and your answer must state that you cannot assist.
        
        {parser.get_format_instructions()}
        
        Context:
        {context}
        """

        # 4. LLM Processing via Chain
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
        chain = llm | parser
        
        result = chain.invoke([
            ("system", system_prompt),
            ("human", request.question)
        ])
        
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
