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

1. **STRICT CONTEXT ADHERENCE**: 
   - Every single word in your output must be directly traceable to the provided context
   - If information appears in the context for a DIFFERENT project, DO NOT use it
   - Each proposal section (Admin Panel, App, Pricing) belongs to ONE specific project only

2. **PRICING ACCURACY (CRITICAL)**:
   - ONLY use pricing from the EXACT matching proposal in the context
   - DO NOT mix pricing details from different proposals
   - DO NOT add payment schedules, discounts, or terms unless they appear in the SAME proposal section
   - If pricing section says "Total: $1210 USD", output EXACTLY that - no payment breakdowns unless explicitly stated

3. **COMPLETE FEATURE EXTRACTION**: 
   - Include ALL numbered features from each relevant section
   - Do not skip, summarize, or combine features
   - Each numbered point in context = one numbered point in output

4. **NO CROSS-CONTAMINATION**:
   - Features from "Food Delivery Platform" should NEVER appear in "Ecommerce Website" proposal
   - Pricing from one proposal should NEVER leak into another
   - When you see multiple proposals in context, use ONLY the one matching the requirements

5. **ZERO ASSUMPTIONS**:
   - Do not add generic features ("user-friendly interface", "responsive design") unless explicitly stated
   - Do not infer technical details not mentioned
   - Do not add payment terms, timelines, or conditions not in the context

6. **SECTION ISOLATION**:
   - Each section (Admin Panel, App, Website, Pricing) should come from the SAME proposal
   - Before including any detail, verify it's from the correct proposal title in the context

=== FORMATTING GUIDELINES ===

1. **Structure**:
   ```
   [Project Title from Context]
   
   Admin Panel Features:
   1. **Feature Name**: Description in 1-2 sentences
   2. **Feature Name**: Description in 1-2 sentences
   
   App Features:
   1. **Feature Name**: Description in 1-2 sentences
   
   Pricing:
   1. Design: $XXX USD
   2. Development: $XXX USD
   3. Total: $XXX USD
   ```

2. **Feature Descriptions**: 
   - Start with bold feature name from context
   - Follow with concise explanation (1-2 sentences)
   - Example: "**User Management**: Administrators can view, edit, and manage all user accounts with the ability to activate, deactivate, or block users."

3. **Pricing Format**:
   - List each cost item exactly as shown in context
   - Use exact numbers and currency from context
   - Do NOT add payment schedules unless they appear in the SAME pricing section

4. **What NOT to Include**:
   - NO introductory paragraphs
   - NO conclusions or "we look forward to" statements
   - NO mixed content from other proposals
   - NO generic marketing language
   - NO payment terms from other proposals

=== CONTEXT FROM KNOWLEDGE BASE ===
{context}

=== PROJECT REQUIREMENTS ===
{requirements}

=== VERIFICATION CHECKLIST (Follow mentally before responding) ===
Before generating output, verify:
1. ✓ Is this feature from the CORRECT proposal matching the requirements?
2. ✓ Is this pricing from the SAME proposal as the features?
3. ✓ Have I included ALL numbered features from relevant sections?
4. ✓ Did I add ANY information not explicitly in the context?
5. ✓ Are there any payment terms that came from a different proposal?

=== OUTPUT ===
Generate the proposal below, ensuring complete accuracy and zero hallucination:
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
    IMPROVED: Keeps entire proposals together to prevent cross-contamination
    
    Strategy:
    1. Split document into complete proposals (each platform/product)
    2. Within each proposal, chunk by sections (Admin Panel, App, Pricing)
    3. Never mix content from different proposals
    4. Add strong proposal boundaries in metadata
    """
    
    text = text.strip()
    
    # More precise proposal detection
    proposal_pattern = r'\n(?=[A-Z][a-zA-Z\s]{3,}(?:Platform|Website|App|System|Portal|Tool|Service|CRM|PICSART|SHOPIFY)[:\s]*\n)'
    
    raw_proposals = re.split(proposal_pattern, text)
    proposals = []
    
    for p in raw_proposals:
        p = p.strip()
        # More strict filtering
        if len(p) > 200 and not p.startswith(('Note', 'Payment', 'Discount')):
            proposals.append(p)
    
    print(f"📋 Identified {len(proposals)} distinct proposals")
    
    all_chunks = []
    
    for proposal_idx, proposal_text in enumerate(proposals, start=1):
        # Extract proposal title more carefully
        title_lines = proposal_text.split('\n')[:3]
        proposal_title = None
        
        for line in title_lines:
            line = line.strip()
            if any(keyword in line for keyword in ['Platform', 'Website', 'App', 'System', 'Portal', 'Tool', 'Service', 'CRM']):
                proposal_title = line.replace('>', '').replace(':', '').strip()
                break
        
        if not proposal_title:
            proposal_title = f"Proposal {proposal_idx}"
        
        print(f"  📄 Processing: {proposal_title}")
        
        # Split by major sections - more comprehensive list
        section_pattern = r'\n(?=>?\s*(?:Admin Panel|App|Website|Landing Page|Development|Pricing|Costing|Technologies|Business Requirements|User Management|Vendor Management|Mobile Application|Main Website|Web Based App|Webpage)[:\s])'
        sections = re.split(section_pattern, proposal_text)
        
        for section_idx, section in enumerate(sections):
            section = section.strip()
            if len(section) < 30:
                continue
            
            # Extract section name
            section_name_match = re.match(r'>?\s*([^:\n]+)[:\n]', section)
            section_name = section_name_match.group(1).strip() if section_name_match else "General"
            
            # Special handling for Pricing sections - keep them complete
            if any(keyword in section_name.lower() for keyword in ['pricing', 'costing', 'estimate']):
                # Don't chunk pricing sections - keep them whole
                doc = Document(
                    page_content=f"=== {proposal_title} - {section_name} ===\n\n{section}",
                    metadata={
                        "proposal_id": proposal_idx,
                        "proposal_title": proposal_title,
                        "section_name": section_name,
                        "section_type": "pricing",
                        "is_complete_section": True,
                        "token_count": count_tokens(section),
                        "chunk_id": 1,
                        "total_chunks": 1
                    }
                )
                all_chunks.append(doc)
                print(f"    💰 Pricing section kept complete: {count_tokens(section)} tokens")
                continue
            
            # For other sections, use token-aware chunking
            section_chunks = chunk_by_features_with_tokens(
                section, 
                max_tokens=1200,  # Slightly larger for more complete features
                overlap_tokens=200,
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
                        "section_type": "features",
                        "section_id": section_idx,
                        "chunk_id": chunk_idx,
                        "total_chunks": len(section_chunks),
                        "token_count": chunk_data['tokens'],
                        "feature_count": chunk_data['features'],
                        "is_complete_section": chunk_idx == 1 and len(section_chunks) == 1
                    }
                )
                all_chunks.append(doc)
            
            print(f"    ✓ {section_name}: {len(section_chunks)} chunks")
    
    total_tokens = sum(count_tokens(doc.page_content) for doc in all_chunks)
    print(f"✅ Created {len(all_chunks)} chunks | Total tokens: {total_tokens:,}")
    
    return all_chunks


def chunk_by_features_with_tokens(text: str, max_tokens: int = 1200, overlap_tokens: int = 200, 
                                   proposal_title: str = "", section_name: str = ""):
    """Chunk text by complete features while respecting token limits"""
    
    header = f"\n=== {proposal_title} - {section_name} ===\n\n"
    header_tokens = count_tokens(header)
    effective_max = max_tokens - header_tokens - 50
    
    # Split by numbered items
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
        
        if feature_tokens > effective_max:
            if current_features:
                chunks.append({
                    'text': current_chunk.strip(),
                    'tokens': current_tokens,
                    'features': len(current_features)
                })
            
            sub_chunks = split_large_feature_by_sentences(feature, effective_max, overlap_tokens, header)
            chunks.extend(sub_chunks)
            
            current_chunk = header
            current_tokens = header_tokens
            current_features = []
            overlap_buffer = []
            continue
        
        if current_tokens + feature_tokens > effective_max and current_features:
            chunks.append({
                'text': current_chunk.strip(),
                'tokens': current_tokens,
                'features': len(current_features)
            })
            
            overlap_text = create_overlap(overlap_buffer, overlap_tokens)
            current_chunk = header + overlap_text + "\n\n" + feature
            current_tokens = header_tokens + count_tokens(overlap_text) + feature_tokens
            current_features = [feature]
            overlap_buffer = [feature]
        else:
            current_chunk += "\n\n" + feature
            current_tokens += feature_tokens
            current_features.append(feature)
            overlap_buffer.append(feature)
            
            if len(overlap_buffer) > 3:
                overlap_buffer.pop(0)
    
    if current_features:
        chunks.append({
            'text': current_chunk.strip(),
            'tokens': current_tokens,
            'features': len(current_features)
        })
    
    return chunks


def split_large_feature_by_sentences(feature: str, max_tokens: int, overlap_tokens: int, header: str):
    """Split a single large feature by sentences"""
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
                overlap = get_last_n_tokens(current, overlap_tokens)
                current = header + overlap + " " + sentence
                current_tokens = count_tokens(current)
            else:
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
    """Create overlap text from recent features"""
    if not features_buffer:
        return ""
    
    overlap_text = ""
    overlap_tokens = 0
    
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
    """Initialize vectorstore and LLM"""
    global vectorstore, llm, is_initialized, initialization_error, tokenizer
    
    try:
        print("📄 Starting initialization...")
        
        tokenizer = tiktoken.get_encoding("cl100k_base")
        print("✅ Tokenizer initialized")
        
        # Lower temperature for stricter adherence
        llm = ChatOpenAI(
            model="gpt-4o-mini", 
            temperature=0.05,  # Even lower for pricing accuracy
            api_key=openai_api_key
        )
        print("✅ LLM initialized")
        
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large", 
            api_key=openai_api_key
        )
        print("✅ Embeddings initialized")
        
        # Use new version to force rebuild with fixed chunking
        persist_dir = "./chroma_store_v3"
        
        if os.path.exists(persist_dir):
            print("📦 Loading existing vectorstore...")
            vectorstore = Chroma(
                embedding_function=embeddings,
                persist_directory=persist_dir
            )
            print("✅ Vectorstore loaded")
        elif os.path.exists(DOC_PATH):
            print(f"📄 Loading knowledge base from {DOC_PATH}")
            loader = Docx2txtLoader(DOC_PATH)
            docs = loader.load()
            
            full_text = "\n".join([d.page_content for d in docs])
            
            chunked_docs = intelligent_chunk_proposals_token_aware(full_text)
            
            print(f"✅ Created {len(chunked_docs)} chunks with improved isolation")
            
            vectorstore = Chroma.from_documents(
                chunked_docs,
                embeddings,
                persist_directory=persist_dir
            )
            print("✅ Chroma vectorstore created")
        else:
            print(f"⚠️ Warning: {DOC_PATH} not found")
            vectorstore = Chroma(
                embedding_function=embeddings,
                persist_directory=persist_dir
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
            raise HTTPException(status_code=503, detail="Service is still initializing.")
        
        print(f"🟢 Query: {query.question}")

        intent = extract_intent(query.question)
        print(f"🧩 Intent: {intent}")

        requirements = format_requirements(intent, query.question)

        # Enhanced retrieval with filtering
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 40,
                "fetch_k": 70,
                "lambda_mult": 0.75  # Higher for more relevance focus
            }
        )
        related_docs = retriever.invoke(query.question)

        if not related_docs:
            raise HTTPException(status_code=404, detail="No related context found.")

        # Group by proposal with strict isolation
        proposals_found = {}
        
        for doc in related_docs:
            proposal_title = doc.metadata.get('proposal_title', 'Unknown')
            section_name = doc.metadata.get('section_name', 'General')
            
            if proposal_title not in proposals_found:
                proposals_found[proposal_title] = {
                    'sections': {},
                    'total_chunks': 0,
                    'relevance_score': 0
                }
            
            if section_name not in proposals_found[proposal_title]['sections']:
                proposals_found[proposal_title]['sections'][section_name] = []
            
            proposals_found[proposal_title]['sections'][section_name].append(doc.page_content)
            proposals_found[proposal_title]['total_chunks'] += 1
        
        # Find the most relevant proposal (most chunks retrieved)
        if proposals_found:
            primary_proposal = max(proposals_found.items(), key=lambda x: x[1]['total_chunks'])[0]
            print(f"🎯 Primary proposal identified: {primary_proposal}")
            
            # Build context from ONLY the primary proposal
            context_parts = []
            context_parts.append(f"\n{'='*80}\n PRIMARY PROPOSAL: {primary_proposal}\n{'='*80}\n")
            
            for section_name, chunks in proposals_found[primary_proposal]['sections'].items():
                section_header = f"\n--- {section_name} ---\n"
                section_content = "\n\n".join(chunks)
                context_parts.append(section_header + section_content)
            
            context = "\n\n".join(context_parts)
        else:
            context = "\n\n".join([doc.page_content for doc in related_docs])

        print(f"📄 Using {proposals_found[primary_proposal]['total_chunks']} chunks from: {primary_proposal}")

        filled_prompt = proposal_prompt.format(context=context, requirements=requirements)
        response = llm.invoke(filled_prompt)

        # Clean output
        cleaned = response.content
        cleaned = re.sub(r'\[Information[^\]]*\]', '', cleaned)
        cleaned = re.sub(r'\[Not available[^\]]*\]', '', cleaned)
        cleaned = re.sub(r'\[Note:.*?\]', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'(?i)(##?\s*(Conclusion|Summary|Final Thoughts).*?)(?=##|$)', '', cleaned, flags=re.DOTALL)
        cleaned = re.sub(r'(?i)(This proposal.*|We look forward.*|Thank you for.*|Feel free.*|Please don\'t.*)$', '', cleaned, flags=re.DOTALL)
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        cleaned = re.sub(r' {2,}', ' ', cleaned)
        cleaned = cleaned.strip()

        print("✅ Proposal generated")
        return {"answer": cleaned, "extracted_intent": intent, "primary_proposal": primary_proposal}

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)