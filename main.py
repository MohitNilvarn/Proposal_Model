from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.documents import Document

import os, re, json
from dotenv import load_dotenv
import asyncio

# ========== LOAD ENV ==========
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("Missing OPENAI_API_KEY in .env")

# ========== WORD CONFIG ==========
DOC_PATH = "Proposal/Proposal Knowledge Base.docx"

# ========== FASTAPI APP ==========
app = FastAPI(title="Proposal Generator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== MODELS ==========
class Query(BaseModel):
    question: str


# ========== PROMPTS ==========
intent_extraction_prompt = """You are an expert at extracting structured information from proposal requests.

Analyze the user's request and extract the following details as JSON:
1. Platforms (Website, App, Admin Panel, etc.)
2. Vendor Type (Single Vendor, Multi Vendor)
3. Project Types (Software Development, AI Automations)
4. Automations (Manychats, CRM, AI Calling)
5. Platform Type (Property Listing, Ecommerce, etc.)
6. Technology (Shopify, Next Js, Wordpress)
7. Services (UI UX, Website Development, App Development, etc.)

Return only JSON, nothing else.

Example:
{
  "platforms": ["Website", "Admin Panel"],
  "vendor_type": "Single Vendor",
  "project_types": ["Software Development"],
  "automations": null,
  "platform_type": "Property Listing",
  "tech_stack": "Next Js",
  "services": ["UI UX Designs", "Website Development"]
}
"""

proposal_prompt_template = """You are a professional proposal writer for AppSynergies. Use ONLY the context provided below to generate a comprehensive project proposal.

STRICT ANTI-HALLUCINATION RULES:
1. ONLY use features, specifications, and details explicitly mentioned in the CONTEXT below
2. NEVER invent or add features not present in the context
3. NEVER add technical details, timelines, or capabilities not stated in the context
4. If information is not in the context, skip that section entirely - DO NOT make assumptions
5. Use EXACT feature names and terminology from the context
6. DO NOT add generic software development knowledge or common industry practices

WRITING STYLE (based on AppSynergies proposal format):
1. Convert each bullet point into a clear, professional description (1-2 sentences)
2. Each feature description should follow this pattern:
   - Start with the feature name in bold
   - Follow with a concise explanation of what it does and its benefit
   - Keep descriptions professional but direct
   - Example: "**User Registration**: A simple and secure registration process using email, phone number, or social media accounts."

3. For each feature, explain:
   - What the feature is
   - How it works (if specified in context)
   - What benefit it provides to users/vendors/admin

4. Maintain professional business tone throughout
5. Use active voice and present tense
6. Keep descriptions concise - aim for 1-2 sentences per feature
7. Group related features under appropriate section headers from the context

FORMATTING RULES:
1. Use section headers exactly as they appear in the context (e.g., "User Registration and Verification:", "Website Features:", "Admin Panel")
2. Present features as numbered lists with bold feature names followed by descriptions
3. DO NOT include "Conclusion", "Summary", or closing statements
4. DO NOT add phrases like "We look forward to..." or "This proposal provides..."
5. End naturally after the last feature section
6. Remove any placeholder text like "[Information not available]"

CRITICAL: Stay strictly within the provided context. If a detail is not explicitly mentioned, do not include it.

---

**CONTEXT FROM KNOWLEDGE BASE:**
{context}

**PROJECT REQUIREMENTS:**
{requirements}

---

Generate the professional proposal below using ONLY information from the context:
"""

intent_prompt = PromptTemplate(
    template=intent_extraction_prompt,
    input_variables=["question"]
)

proposal_prompt = PromptTemplate(
    template=proposal_prompt_template,
    input_variables=["context", "requirements"]
)

# ========== GLOBAL VARIABLES ==========
vectorstore = None
llm = None
is_initialized = False
initialization_error = None


# ========== PRE-SPLIT FUNCTION ==========
def split_proposals(text: str):
    """
    Splits a long proposal document into individual proposals
    using section titles like 'Platform', 'Website', or 'App'.
    Returns a list of proposal strings.
    """
    text = text.strip()
    pattern = r'(?=(?:[A-Z][a-zA-Z ]{2,}\s(?:Platform|Website|App|System)))'
    sections = re.split(pattern, text)
    proposals = [s.strip() for s in sections if len(s.strip()) > 100]
    print(f"🧾 Found {len(proposals)} proposals in document.")
    return proposals


# ========== BACKGROUND INITIALIZATION ==========
def initialize_services():
    """Initialize vectorstore and LLM in background"""
    global vectorstore, llm, is_initialized, initialization_error
    
    try:
        print("📄 Starting initialization...")
        
        # Initialize LLM
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=openai_api_key)
        print("✅ LLM initialized.")
        
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=openai_api_key)
        
        # Check if vectorstore already exists
        if os.path.exists("./chroma_store"):
            print("📦 Loading existing vectorstore...")
            vectorstore = Chroma(
                embedding_function=embeddings,
                persist_directory="./chroma_store"
            )
            print("✅ Vectorstore loaded from disk.")
        elif os.path.exists(DOC_PATH):
            print(f"📄 Loading knowledge base from {DOC_PATH}")
            loader = Docx2txtLoader(DOC_PATH)
            docs = loader.load()
            
            full_text = "\n".join([d.page_content for d in docs])
            
            # Step 1: Split into proposals
            proposals = split_proposals(full_text)
            
            # Step 2: Initialize text splitter
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,
                chunk_overlap=350,
                separators=["\n\n", "\n", "•", "-", ":", ". "],
                length_function=len
            )
            
            # Step 3: Chunk each proposal separately
            chunked_docs = []
            for idx, proposal_text in enumerate(proposals, start=1):
                chunks = splitter.split_text(proposal_text)
                for i, chunk in enumerate(chunks, start=1):
                    chunked_docs.append(
                        Document(
                            page_content=chunk,
                            metadata={"proposal_id": idx, "chunk_id": i}
                        )
                    )
            
            print(f"✅ Created {len(chunked_docs)} total chunks from {len(proposals)} proposals.")
            
            vectorstore = Chroma.from_documents(
                chunked_docs,
                embeddings,
                persist_directory="./chroma_store"
            )
            print("✅ Chroma vectorstore created and persisted.")
        else:
            print(f"⚠️ Warning: {DOC_PATH} not found. Creating empty vectorstore.")
            vectorstore = Chroma(
                embedding_function=embeddings,
                persist_directory="./chroma_store"
            )
        
        is_initialized = True
        print("✅ All services initialized successfully!")
        
    except Exception as e:
        initialization_error = str(e)
        print(f"❌ Initialization error: {e}")
        import traceback
        traceback.print_exc()


# ========== STARTUP EVENT ==========
@app.on_event("startup")
async def startup_event():
    """Start initialization in background - don't block startup"""
    print("🚀 FastAPI starting up...")
    import threading
    thread = threading.Thread(target=initialize_services, daemon=True)
    thread.start()
    print("🔄 Initialization running in background...")


# ========== HELPER FUNCTIONS ==========
def extract_intent(question: str) -> dict:
    try:
        filled_prompt = intent_prompt.format(question=question)
        response = llm.invoke(filled_prompt)
        content = response.content.strip()
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            return json.loads(match.group())
        return {}
    except Exception as e:
        print("❌ Error extracting intent:", e)
        return {}

def format_requirements(intent: dict, question: str) -> str:
    parts = [f"**Original Request:** {question}\n"]
    if intent.get("platform_type"):
        parts.append(f"**Platform Type:** {intent['platform_type']}")
    if intent.get("platforms"):
        parts.append(f"**Platforms:** {', '.join(intent['platforms'])}")
    if intent.get("vendor_type"):
        parts.append(f"**Vendor Type:** {intent['vendor_type']}")
    if intent.get("project_types"):
        parts.append(f"**Project Type:** {', '.join(intent['project_types'])}")
    if intent.get("automations"):
        parts.append(f"**Automations:** {intent['automations']}")
    if intent.get("tech_stack"):
        parts.append(f"**Technology Stack:** {intent['tech_stack']}")
    if intent.get("services"):
        parts.append(f"**Services:** {', '.join(intent['services'])}")
    return "\n".join(parts)


# ========== ROUTES ==========
@app.get("/")
def root():
    status = "initializing" if not is_initialized else "ready"
    return {
        "status": "online", 
        "message": "Proposal Generator API is running.",
        "initialization_status": status
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "initialized": is_initialized,
        "error": initialization_error
    }

@app.get("/status")
def status():
    return {
        "initialized": is_initialized,
        "error": initialization_error,
        "vectorstore_ready": vectorstore is not None,
        "llm_ready": llm is not None
    }

@app.post("/ask")
def ask(query: Query):
    try:
        if initialization_error:
            raise HTTPException(status_code=500, detail=f"Initialization failed: {initialization_error}")
        
        if not is_initialized or vectorstore is None or llm is None:
            raise HTTPException(status_code=503, detail="Service is still initializing. Please try again in a few moments. Check /status endpoint.")
        
        print(f"🟢 Received query: {query.question}")

        intent = extract_intent(query.question)
        print("🧩 Extracted Intent:", intent)

        requirements = format_requirements(intent, query.question)

        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 15}
        )
        related_docs = retriever.invoke(query.question)

        if not related_docs:
            raise HTTPException(status_code=404, detail="No related context found in document.")

        seen_content = set()
        unique_docs = []
        for doc in related_docs:
            content = doc.page_content.strip()
            if content and content not in seen_content:
                seen_content.add(content)
                unique_docs.append(content)
        
        context = "\n\n---SECTION---\n\n".join(unique_docs)

        filled_prompt = proposal_prompt.format(context=context, requirements=requirements)
        response = llm.invoke(filled_prompt)

        cleaned = response.content
        cleaned = re.sub(r'\[Information[^\]]*\]', '', cleaned)
        cleaned = re.sub(r'\[Not available[^\]]*\]', '', cleaned)
        cleaned = re.sub(r'\[Note:.*?\]', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'(?i)(##?\s*(Conclusion|Summary|Final Thoughts|Next Steps|In Summary).*?)(?=##|$)', '', cleaned, flags=re.DOTALL)
        cleaned = re.sub(r'(?i)(This proposal (provides|outlines|presents).*|We look forward.*|Thank you for considering.*|Feel free to reach out.*|Please don\'t hesitate.*)$', '', cleaned, flags=re.DOTALL)
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        cleaned = re.sub(r' {2,}', ' ', cleaned)
        cleaned = cleaned.strip()

        print("✅ Proposal generated successfully.")
        return {"answer": cleaned, "extracted_intent": intent}

    except HTTPException:
        raise
    except Exception as e:
        print("❌ Error in /ask:", str(e))
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal Error: {str(e)}")


# ========== MAIN ==========
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
