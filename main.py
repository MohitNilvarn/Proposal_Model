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
app = FastAPI(title="Proposal Generator API - Multi-Agent")

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


# ========== AGENT 1: RETRIEVER AGENT PROMPTS ==========
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

retriever_agent_prompt = """You are Agent 1: The Retriever Agent. Your ONLY job is to extract exact bullet points from the provided context.

=== YOUR MISSION ===
Extract ALL relevant features, specifications, and details from the context below in a structured bullet-point format.

=== CRITICAL RULES ===
1. **EXACT EXTRACTION ONLY**: Copy bullet points/features EXACTLY as they appear in the context
2. **NO ELABORATION**: Do not add descriptions, explanations, or additional sentences
3. **NO HALLUCINATION**: Every single bullet point must exist in the context
4. **COMPLETE EXTRACTION**: Include ALL numbered points from each relevant section
5. **PRESERVE STRUCTURE**: Maintain section headers (Admin Panel, App, Website, Pricing)
6. **PRESERVE NUMBERING**: Keep the original numbering from the context

=== OUTPUT FORMAT ===
Return a structured list following this exact format:

```
PROPOSAL: [Exact proposal title from context]

ADMIN PANEL:
1. [Exact feature name/text from context]
2. [Exact feature name/text from context]
3. [Exact feature name/text from context]

APP FEATURES:
1. [Exact feature name/text from context]
2. [Exact feature name/text from context]

WEBSITE FEATURES:
1. [Exact feature name/text from context]

PRICING:
1. [COMPLETE pricing line with amount - e.g., "Design: ₹2,00,000"]
2. [COMPLETE pricing line with amount - e.g., "Development: ₹4,35,000"]
3. [COMPLETE pricing line with amount - e.g., "Total: ₹7,54,000"]
```

=== WHAT NOT TO DO ===
❌ Do NOT write: "User Management: Allows administrators to manage users efficiently"
✅ DO write: "User Management"

❌ Do NOT write: "Design" (missing price)
✅ DO write: "Design: ₹2,00,000"

❌ Do NOT add explanations or descriptions
✅ DO extract only the feature names/titles with exact pricing amounts

❌ Do NOT combine or summarize features
✅ DO list each feature separately

=== CONTEXT FROM KNOWLEDGE BASE ===
{context}

=== PROJECT REQUIREMENTS ===
{requirements}

=== YOUR OUTPUT (Extract bullet points below) ===
"""


# ========== AGENT 2: WRITER AGENT PROMPTS ==========
writer_agent_prompt = """You are Agent 2: The Writer Agent. Your job is to elaborate the bullet points provided by Agent 1 into professional descriptions.

=== YOUR MISSION ===
Take each bullet point from Agent 1 and expand it into a clear, professional 1-2 sentence description.

=== CRITICAL ANTI-HALLUCINATION RULES ===
1. **ONLY ELABORATE GIVEN BULLETS**: Work ONLY with the bullet points provided by Agent 1
2. **NO NEW FEATURES**: Do not add features not in Agent 1's list
3. **STAY ON TOPIC**: Describe only what the bullet point name suggests - do not invent details
4. **USE CONTEXT**: Refer to the original context to ensure accuracy of your descriptions
5. **GENERIC IS OKAY**: If context lacks details, write a generic but accurate description based on the feature name
6. **NO ASSUMPTIONS**: Do not add technical specs, timelines, or implementation details not in context

=== CRITICAL: PRICING MUST BE EXACT ===
**FOR PRICING SECTIONS - ABSOLUTE RULE:**
- Copy pricing numbers EXACTLY as they appear in Agent 1's output
- Do NOT convert currencies (₹ to USD or vice versa)
- Do NOT change amounts
- Do NOT add your own pricing breakdowns
- If Agent 1 says "Design: ₹2,00,000", you write "**Design**: ₹2,00,000"
- Only add a brief 1-sentence description AFTER the exact price
- Example: "**Design**: ₹2,00,000 - Professional UI/UX design services for the platform."

=== ELABORATION GUIDELINES ===

For FEATURE bullet points, write:
- **[Feature Name]**: [1-2 sentence description explaining what it does and its benefit]

For PRICING bullet points, write:
- **[Item Name]**: [EXACT PRICE FROM AGENT 1] - [Brief 1-sentence description]

Example transformations:

**Features:**
Input: "User Management"
Output: "**User Management**: Administrators can view, edit, and manage all user accounts with the ability to activate, deactivate, or block users based on their activity and policy compliance."

**Pricing (CRITICAL - PRESERVE EXACT NUMBERS):**
Input: "Design: ₹2,00,000"
Output: "**Design**: ₹2,00,000 - Professional UI/UX design services tailored to the platform's requirements."

Input: "Total: ₹7,54,000"
Output: "**Total**: ₹7,54,000 - The complete project cost including development, testing, deployment, and annual maintenance."

DO NOT CONVERT: If Agent 1 says ₹2,00,000, DO NOT write $2,500 or any other amount.

=== DESCRIPTION FORMULA ===
1. Start with what the feature does (action/capability)
2. Add the benefit or purpose
3. Keep it concise (1-2 sentences max)
4. Use professional business language

=== WHAT TO AVOID ===
❌ "This feature will revolutionize your business" (marketing fluff)
❌ "Built using React and Node.js" (tech details not in context)
❌ "Launching in Q1 2024" (timeline assumptions)
❌ "The most advanced system available" (subjective claims)
❌ Converting "₹2,00,000" to "$2,500" or any other currency/amount (NEVER CHANGE PRICING)
❌ Writing "Design: Approximately ₹2,00,000" (remove words like "approximately", use exact amounts)
❌ Rounding or approximating numbers (preserve exact amounts from Agent 1)

=== ORIGINAL CONTEXT (for reference only) ===
{context}

=== BULLET POINTS FROM AGENT 1 ===
{bullet_points}

=== YOUR OUTPUT (Elaborate each bullet point below) ===
"""

# ========== PROMPT TEMPLATES ==========
intent_prompt = PromptTemplate(
    template=intent_extraction_prompt,
    input_variables=["question"]
)

retriever_prompt = PromptTemplate(
    template=retriever_agent_prompt,
    input_variables=["context", "requirements"]
)

writer_prompt = PromptTemplate(
    template=writer_agent_prompt,
    input_variables=["context", "bullet_points"]
)

# ========== GLOBAL VARIABLES ==========
vectorstore = None
retriever_agent = None  # Agent 1
writer_agent = None      # Agent 2
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
    """
    
    text = text.strip()
    
    proposal_pattern = r'\n(?=[A-Z][a-zA-Z\s]{3,}(?:Platform|Website|App|System|Portal|Tool|Service|CRM|PICSART|SHOPIFY)[:\s]*\n)'
    
    raw_proposals = re.split(proposal_pattern, text)
    proposals = []
    
    for p in raw_proposals:
        p = p.strip()
        if len(p) > 200 and not p.startswith(('Note', 'Payment', 'Discount')):
            proposals.append(p)
    
    print(f"📋 Identified {len(proposals)} distinct proposals")
    
    all_chunks = []
    
    for proposal_idx, proposal_text in enumerate(proposals, start=1):
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
        
        section_pattern = r'\n(?=>?\s*(?:Admin Panel|App|Website|Landing Page|Development|Pricing|Costing|Technologies|Business Requirements|User Management|Vendor Management|Mobile Application|Main Website|Web Based App|Webpage)[:\s])'
        sections = re.split(section_pattern, proposal_text)
        
        for section_idx, section in enumerate(sections):
            section = section.strip()
            if len(section) < 30:
                continue
            
            section_name_match = re.match(r'>?\s*([^:\n]+)[:\n]', section)
            section_name = section_name_match.group(1).strip() if section_name_match else "General"
            
            # Keep pricing sections complete
            if any(keyword in section_name.lower() for keyword in ['pricing', 'costing', 'estimate']):
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
                print(f"    💰 Pricing section: {count_tokens(section)} tokens")
                continue
            
            section_chunks = chunk_by_features_with_tokens(
                section, 
                max_tokens=1200,
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
    """Initialize vectorstore and both agents"""
    global vectorstore, retriever_agent, writer_agent, is_initialized, initialization_error, tokenizer
    
    try:
        print("📄 Starting multi-agent initialization...")
        
        tokenizer = tiktoken.get_encoding("cl100k_base")
        print("✅ Tokenizer initialized")
        
        # Agent 1: Retriever (extracts bullet points) - Low temperature for exact extraction
        retriever_agent = ChatOpenAI(
            model="gpt-4o-mini", 
            temperature=0.0,  # Zero creativity - exact extraction only
            api_key=openai_api_key
        )
        print("✅ Agent 1 (Retriever) initialized")
        
        # Agent 2: Writer (elaborates bullet points) - Very low temperature for pricing accuracy
        writer_agent = ChatOpenAI(
            model="gpt-4o-mini", 
            temperature=0.1,  # Lower creativity - must preserve exact pricing
            api_key=openai_api_key
        )
        print("✅ Agent 2 (Writer) initialized")
        
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large", 
            api_key=openai_api_key
        )
        print("✅ Embeddings initialized")
        
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
            
            print(f"✅ Created {len(chunked_docs)} chunks")
            
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
        print("✅ Multi-agent system initialized successfully!")
        
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
        response = retriever_agent.invoke(filled_prompt)
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
        "message": "Multi-Agent Proposal Generator API",
        "initialization_status": status,
        "agents": {
            "agent_1": "Retriever (extracts bullet points)",
            "agent_2": "Writer (elaborates descriptions)"
        }
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
        "retriever_agent_ready": retriever_agent is not None,
        "writer_agent_ready": writer_agent is not None
    }

@app.post("/ask")
def ask(query: Query):
    try:
        if initialization_error:
            raise HTTPException(status_code=500, detail=f"Initialization failed: {initialization_error}")
        
        if not is_initialized or vectorstore is None or retriever_agent is None or writer_agent is None:
            raise HTTPException(status_code=503, detail="Service is still initializing.")
        
        print(f"\n{'='*80}")
        print(f"🟢 NEW REQUEST: {query.question}")
        print(f"{'='*80}\n")

        # Step 1: Extract intent
        intent = extract_intent(query.question)
        print(f"🧩 Intent extracted: {intent.get('platform_type', 'Unknown')}")

        requirements = format_requirements(intent, query.question)

        # Step 2: Retrieve relevant context
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 40,
                "fetch_k": 70,
                "lambda_mult": 0.75
            }
        )
        related_docs = retriever.invoke(query.question)

        if not related_docs:
            raise HTTPException(status_code=404, detail="No related context found.")

        # Step 3: Identify primary proposal
        proposals_found = {}
        
        for doc in related_docs:
            proposal_title = doc.metadata.get('proposal_title', 'Unknown')
            section_name = doc.metadata.get('section_name', 'General')
            
            if proposal_title not in proposals_found:
                proposals_found[proposal_title] = {
                    'sections': {},
                    'total_chunks': 0
                }
            
            if section_name not in proposals_found[proposal_title]['sections']:
                proposals_found[proposal_title]['sections'][section_name] = []
            
            proposals_found[proposal_title]['sections'][section_name].append(doc.page_content)
            proposals_found[proposal_title]['total_chunks'] += 1
        
        primary_proposal = max(proposals_found.items(), key=lambda x: x[1]['total_chunks'])[0]
        print(f"🎯 Primary proposal: {primary_proposal}")
        
        # Build context from primary proposal only
        context_parts = []
        context_parts.append(f"\n{'='*80}\nPRIMARY PROPOSAL: {primary_proposal}\n{'='*80}\n")
        
        for section_name, chunks in proposals_found[primary_proposal]['sections'].items():
            section_header = f"\n--- {section_name} ---\n"
            section_content = "\n\n".join(chunks)
            context_parts.append(section_header + section_content)
        
        context = "\n\n".join(context_parts)

        print(f"📄 Context: {proposals_found[primary_proposal]['total_chunks']} chunks")

        # ========== AGENT 1: RETRIEVER - Extract bullet points ==========
        print(f"\n🤖 AGENT 1 (Retriever): Extracting bullet points...")
        
        retriever_filled_prompt = retriever_prompt.format(
            context=context, 
            requirements=requirements
        )
        retriever_response = retriever_agent.invoke(retriever_filled_prompt)
        bullet_points = retriever_response.content.strip()
        
        print(f"✅ Agent 1 completed: Extracted {bullet_points.count(chr(10))} lines")
        print(f"\n--- AGENT 1 OUTPUT (Sample) ---")
        print(bullet_points[:500] + "..." if len(bullet_points) > 500 else bullet_points)
        print(f"--- END SAMPLE ---\n")

        # ========== AGENT 2: WRITER - Elaborate bullet points ==========
        print(f"🤖 AGENT 2 (Writer): Elaborating bullet points...")
        
        writer_filled_prompt = writer_prompt.format(
            context=context,
            bullet_points=bullet_points
        )
        writer_response = writer_agent.invoke(writer_filled_prompt)
        elaborated_proposal = writer_response.content.strip()
        
        print(f"✅ Agent 2 completed: Generated proposal")

        # Clean output
        cleaned = elaborated_proposal
        cleaned = re.sub(r'\[Information[^\]]*\]', '', cleaned)
        cleaned = re.sub(r'\[Not available[^\]]*\]', '', cleaned)
        cleaned = re.sub(r'\[Note:.*?\]', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        cleaned = re.sub(r' {2,}', ' ', cleaned)
        cleaned = cleaned.strip()

        print(f"\n✅ PROPOSAL GENERATED SUCCESSFULLY")
        print(f"{'='*80}\n")

        return {
            "answer": cleaned,
            "extracted_intent": intent,
            "primary_proposal": primary_proposal,
            "agent_1_output": bullet_points,  # Include for debugging
            "processing_steps": {
                "step_1": "Intent extraction completed",
                "step_2": f"Retrieved {len(related_docs)} chunks",
                "step_3": f"Identified primary proposal: {primary_proposal}",
                "step_4": "Agent 1 extracted bullet points",
                "step_5": "Agent 2 elaborated descriptions"
            }
        }

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