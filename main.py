from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.documents import Document
import tiktoken

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

proposal_prompt_template = """You are a professional proposal writer for AppSynergies. Your task is to generate a comprehensive project proposal using ONLY the information provided in the context below.

=== CRITICAL ANTI-HALLUCINATION RULES ===

1. **ONLY USE CONTEXT**: Every feature, specification, and detail MUST come directly from the provided context. If it's not in the context, DO NOT include it.

2. **COMPLETE EXTRACTION**: Include ALL features mentioned in the context for each section. Do not skip or summarize features - each numbered point in the context should appear in your output.

3. **EXACT TERMINOLOGY**: Use the exact feature names and technical terms from the context. Do not paraphrase or use synonyms.

4. **NO ASSUMPTIONS**: Do not add:
   - Generic software features
   - Industry best practices not mentioned
   - Technical implementation details not specified
   - Timeline or cost estimates not provided
   - Features from your general knowledge

5. **VERIFY COMPLETENESS**: Before finishing, mentally check that you've included every numbered point from each relevant section in the context.

=== FORMATTING GUIDELINES ===

1. **Section Headers**: Use clear headers matching the context (e.g., "Admin Panel Features:", "Mobile Application Features:")

2. **Feature Format**: For each feature, write:
   - **Feature Name**: Brief 1-2 sentence description explaining what it does and its benefit
   - Example: "**User Management**: Administrators can view, edit, and manage all user accounts with the ability to activate, deactivate, or block users based on activity or policy compliance."

3. **Numbering**: Maintain numbered lists exactly as they appear in the context

4. **No Fluff**: Do NOT add:
   - Introductory paragraphs
   - Conclusion sections
   - "We look forward to..." statements
   - "This proposal provides..." summaries
   - Marketing language not in the context

5. **Clean Output**: Remove any placeholder text like "[Information not available]"

=== CONTEXT FROM KNOWLEDGE BASE ===
{context}

=== PROJECT REQUIREMENTS ===
{requirements}

=== INSTRUCTIONS ===
Now generate a comprehensive proposal that:
1. Extracts ALL features from relevant sections in the context
2. Formats each feature professionally with clear descriptions
3. Groups features under appropriate section headers
4. Stays 100% faithful to the source material
5. Ends naturally after the last feature (no conclusion)

Begin the proposal below:
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
tokenizer = None


# ========== TOKEN-AWARE SEMANTIC CHUNKING ==========
def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Count tokens in text using tiktoken"""
    global tokenizer
    if tokenizer is None:
        tokenizer = tiktoken.get_encoding(encoding_name)
    return len(tokenizer.encode(text))


def intelligent_chunk_proposals_token_aware(text: str):
    """
    HYBRID APPROACH: Semantic chunking with token awareness
    
    Benefits:
    1. Respects semantic boundaries (numbered lists, sections)
    2. Uses token counts for precise size control
    3. Optimizes for embedding model limits (8191 tokens for text-embedding-3-large)
    4. Prevents mid-feature splits
    
    Strategy:
    - Max chunk size: 1000 tokens (~750 words) - optimal for embedding quality
    - Overlap: 150 tokens - ensures context continuity
    - Semantic boundaries: Always respected over token limits
    """
    
    text = text.strip()
    
    # Pattern to identify major proposal boundaries
    proposal_pattern = r'(?=(?:^|\n)(?:[A-Z][a-zA-Z\s]{3,}(?:Platform|Website|App|System|Portal|Tool|Service))\s*:?\s*\n)'
    
    raw_proposals = re.split(proposal_pattern, text)
    proposals = [p.strip() for p in raw_proposals if len(p.strip()) > 100]
    
    print(f"📋 Identified {len(proposals)} major proposals")
    
    all_chunks = []
    
    for proposal_idx, proposal_text in enumerate(proposals, start=1):
        # Extract proposal title
        title_match = re.match(r'^([A-Z][a-zA-Z\s]{3,}(?:Platform|Website|App|System|Portal|Tool|Service))', proposal_text)
        proposal_title = title_match.group(1).strip() if title_match else f"Proposal {proposal_idx}"
        
        # Split by major sections
        section_pattern = r'\n(?=>?\s*(?:Admin Panel|App|Website|Landing Page|Development|Pricing|Technologies|Business Requirements|User Management|Vendor Management)[:\s])'
        sections = re.split(section_pattern, proposal_text)
        
        for section_idx, section in enumerate(sections):
            section = section.strip()
            if len(section) < 50:
                continue
            
            # Extract section name
            section_name_match = re.match(r'>?\s*([^:\n]+)[:\n]', section)
            section_name = section_name_match.group(1).strip() if section_name_match else "General"
            
            # Token-aware semantic chunking
            section_chunks = chunk_by_features_with_tokens(
                section, 
                max_tokens=1000,  # Optimal size for embeddings
                overlap_tokens=150,
                proposal_title=proposal_title,
                section_name=section_name
            )
            
            for chunk_idx, chunk_data in enumerate(section_chunks, start=1):
                doc = Document(
                    page_content=chunk_data['text'],
                    metadata={
                        "proposal_id": proposal_idx,
                        "proposal_title": proposal_title,
                        "section_name": section_name,
                        "section_id": section_idx,
                        "chunk_id": chunk_idx,
                        "total_chunks": len(section_chunks),
                        "token_count": chunk_data['tokens'],
                        "feature_count": chunk_data['features']
                    }
                )
                all_chunks.append(doc)
    
    total_tokens = sum(count_tokens(doc.page_content) for doc in all_chunks)
    print(f"✅ Created {len(all_chunks)} chunks | Total tokens: {total_tokens:,}")
    
    return all_chunks


def chunk_by_features_with_tokens(text: str, max_tokens: int = 1000, overlap_tokens: int = 150, 
                                   proposal_title: str = "", section_name: str = ""):
    """
    Chunk text by complete features while respecting token limits.
    
    SEMANTIC PRIORITY:
    1. Keep numbered items together
    2. Don't split in middle of sentences
    3. Maintain feature descriptions intact
    4. Add overlap for context continuity
    
    TOKEN AWARENESS:
    - Use actual token counting (not char approximation)
    - Stay under max_tokens for embedding efficiency
    - Overlap measured in tokens for precision
    """
    
    # Add section header for context
    header = f"\n{proposal_title} - {section_name}\n\n"
    header_tokens = count_tokens(header)
    
    # Adjust max tokens to account for header
    effective_max = max_tokens - header_tokens - 50  # 50 token buffer
    
    # Split by numbered items (features)
    numbered_pattern = r'(?=\n\s*\d+\.?\s+)'
    features = re.split(numbered_pattern, text)
    features = [f.strip() for f in features if f.strip()]
    
    chunks = []
    current_chunk = header
    current_tokens = header_tokens
    current_features = []
    overlap_buffer = []
    
    for feature in features:
        feature_tokens = count_tokens(feature)
        
        # If single feature exceeds limit, split it carefully
        if feature_tokens > effective_max:
            # Save current chunk if exists
            if current_features:
                chunks.append({
                    'text': current_chunk.strip(),
                    'tokens': current_tokens,
                    'features': len(current_features)
                })
            
            # Split large feature by sentences
            sub_chunks = split_large_feature_by_sentences(feature, effective_max, overlap_tokens, header)
            chunks.extend(sub_chunks)
            
            # Reset for next chunk
            current_chunk = header
            current_tokens = header_tokens
            current_features = []
            overlap_buffer = []
            continue
        
        # Check if adding this feature exceeds limit
        if current_tokens + feature_tokens > effective_max and current_features:
            # Finalize current chunk
            chunks.append({
                'text': current_chunk.strip(),
                'tokens': current_tokens,
                'features': len(current_features)
            })
            
            # Start new chunk with overlap
            overlap_text = create_overlap(overlap_buffer, overlap_tokens)
            current_chunk = header + overlap_text + "\n\n" + feature
            current_tokens = header_tokens + count_tokens(overlap_text) + feature_tokens
            current_features = [feature]
            overlap_buffer = [feature]
        else:
            # Add to current chunk
            current_chunk += "\n\n" + feature
            current_tokens += feature_tokens
            current_features.append(feature)
            overlap_buffer.append(feature)
            
            # Keep only last 3 features in overlap buffer
            if len(overlap_buffer) > 3:
                overlap_buffer.pop(0)
    
    # Add final chunk
    if current_features:
        chunks.append({
            'text': current_chunk.strip(),
            'tokens': current_tokens,
            'features': len(current_features)
        })
    
    return chunks


def split_large_feature_by_sentences(feature: str, max_tokens: int, overlap_tokens: int, header: str):
    """Split a single large feature by sentences when it exceeds token limit"""
    
    sentences = re.split(r'(?<=[.!?])\s+', feature)
    chunks = []
    
    current = header
    current_tokens = count_tokens(header)
    
    for sentence in sentences:
        sentence_tokens = count_tokens(sentence)
        
        if current_tokens + sentence_tokens > max_tokens:
            if current != header:
                chunks.append({
                    'text': current.strip(),
                    'tokens': current_tokens,
                    'features': 1
                })
                # Start new with overlap (last part of current)
                overlap = get_last_n_tokens(current, overlap_tokens)
                current = header + overlap + " " + sentence
                current_tokens = count_tokens(current)
            else:
                # Single sentence too large, add as-is
                chunks.append({
                    'text': header + sentence,
                    'tokens': count_tokens(header + sentence),
                    'features': 1
                })
                current = header
                current_tokens = count_tokens(header)
        else:
            current += " " + sentence
            current_tokens += sentence_tokens
    
    if current != header:
        chunks.append({
            'text': current.strip(),
            'tokens': current_tokens,
            'features': 1
        })
    
    return chunks


def create_overlap(features_buffer: list, target_tokens: int) -> str:
    """Create overlap text from recent features, targeting specific token count"""
    if not features_buffer:
        return ""
    
    overlap_text = ""
    overlap_tokens = 0
    
    # Add features from end until we reach target tokens
    for feature in reversed(features_buffer):
        feature_tokens = count_tokens(feature)
        if overlap_tokens + feature_tokens <= target_tokens:
            overlap_text = feature + "\n\n" + overlap_text
            overlap_tokens += feature_tokens
        else:
            break
    
    return overlap_text.strip()


def get_last_n_tokens(text: str, n_tokens: int) -> str:
    """Get the last N tokens from text"""
    global tokenizer
    if tokenizer is None:
        tokenizer = tiktoken.get_encoding("cl100k_base")
    
    tokens = tokenizer.encode(text)
    if len(tokens) <= n_tokens:
        return text
    
    last_tokens = tokens[-n_tokens:]
    return tokenizer.decode(last_tokens)


# ========== BACKGROUND INITIALIZATION ==========
def initialize_services():
    """Initialize vectorstore and LLM in background"""
    global vectorstore, llm, is_initialized, initialization_error, tokenizer
    
    try:
        print("📄 Starting initialization...")
        
        # Initialize tokenizer
        tokenizer = tiktoken.get_encoding("cl100k_base")
        print("✅ Tokenizer initialized")
        
        # Initialize LLM with better parameters for accuracy
        llm = ChatOpenAI(
            model="gpt-4o-mini", 
            temperature=0.1,
            api_key=openai_api_key
        )
        print("✅ LLM initialized")
        
        # Use better embeddings
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large", 
            api_key=openai_api_key
        )
        print("✅ Embeddings initialized")
        
        # Check if vectorstore exists
        if os.path.exists("./chroma_store_v2"):  # New version to force rebuild
            print("📦 Loading existing vectorstore...")
            vectorstore = Chroma(
                embedding_function=embeddings,
                persist_directory="./chroma_store_v2"
            )
            print("✅ Vectorstore loaded from disk")
        elif os.path.exists(DOC_PATH):
            print(f"📄 Loading knowledge base from {DOC_PATH}")
            loader = Docx2txtLoader(DOC_PATH)
            docs = loader.load()
            
            full_text = "\n".join([d.page_content for d in docs])
            
            # Use token-aware intelligent chunking
            chunked_docs = intelligent_chunk_proposals_token_aware(full_text)
            
            print(f"✅ Created {len(chunked_docs)} chunks with token awareness")
            
            # Show sample metadata
            if chunked_docs:
                sample = chunked_docs[0]
                print(f"📊 Sample chunk: {sample.metadata['token_count']} tokens, "
                      f"{sample.metadata['feature_count']} features")
            
            vectorstore = Chroma.from_documents(
                chunked_docs,
                embeddings,
                persist_directory="./chroma_store_v2"
            )
            print("✅ Chroma vectorstore created and persisted")
        else:
            print(f"⚠️ Warning: {DOC_PATH} not found")
            vectorstore = Chroma(
                embedding_function=embeddings,
                persist_directory="./chroma_store_v2"
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

        # Enhanced retrieval with MMR
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 35,  # More chunks for completeness
                "fetch_k": 60,
                "lambda_mult": 0.7
            }
        )
        related_docs = retriever.invoke(query.question)

        if not related_docs:
            raise HTTPException(status_code=404, detail="No related context found in document.")

        # Group by proposal and section
        chunks_by_section = {}
        for doc in related_docs:
            proposal_title = doc.metadata.get('proposal_title', 'Unknown')
            section_name = doc.metadata.get('section_name', 'General')
            key = f"{proposal_title}::{section_name}"
            
            if key not in chunks_by_section:
                chunks_by_section[key] = {
                    'chunks': [],
                    'total_tokens': 0,
                    'total_features': 0
                }
            
            chunks_by_section[key]['chunks'].append(doc.page_content)
            chunks_by_section[key]['total_tokens'] += doc.metadata.get('token_count', 0)
            chunks_by_section[key]['total_features'] += doc.metadata.get('feature_count', 0)
        
        # Reconstruct context
        context_parts = []
        for key, data in chunks_by_section.items():
            proposal_title, section_name = key.split("::")
            section_header = f"\n{'='*60}\n{proposal_title} - {section_name}\n{'='*60}\n"
            section_content = "\n\n".join(data['chunks'])
            context_parts.append(section_header + section_content)
        
        context = "\n\n".join(context_parts)

        total_context_tokens = sum(data['total_tokens'] for data in chunks_by_section.values())
        total_features = sum(data['total_features'] for data in chunks_by_section.values())
        
        print(f"📄 Retrieved {len(related_docs)} chunks from {len(chunks_by_section)} sections")
        print(f"📊 Context: ~{total_context_tokens:,} tokens, ~{total_features} features")

        filled_prompt = proposal_prompt.format(context=context, requirements=requirements)
        response = llm.invoke(filled_prompt)

        # Clean up output
        cleaned = response.content
        cleaned = re.sub(r'\[Information[^\]]*\]', '', cleaned)
        cleaned = re.sub(r'\[Not available[^\]]*\]', '', cleaned)
        cleaned = re.sub(r'\[Note:.*?\]', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'(?i)(##?\s*(Conclusion|Summary|Final Thoughts|Next Steps|In Summary).*?)(?=##|$)', '', cleaned, flags=re.DOTALL)
        cleaned = re.sub(r'(?i)(This proposal (provides|outlines|presents).*|We look forward.*|Thank you for considering.*|Feel free to reach out.*|Please don\'t hesitate.*)$', '', cleaned, flags=re.DOTALL)
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        cleaned = re.sub(r' {2,}', ' ', cleaned)
        cleaned = cleaned.strip()

        print("✅ Proposal generated successfully")
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