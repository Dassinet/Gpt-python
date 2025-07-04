from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, Body, Request, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
import json
import os
import asyncio
from typing import List, Dict, Any, Optional, Union
import shutil
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import time
from io import BytesIO
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from contextlib import asynccontextmanager

from storage import CloudflareR2Storage
from rag import EnhancedRAG, CLAUDE_AVAILABLE, BM25_AVAILABLE, GEMINI_AVAILABLE, GROQ_AVAILABLE
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

load_dotenv()

# Define the cleanup function first
async def cleanup_r2_expired_files():
    """Periodic task to clean up expired R2 files"""
    print("Running scheduled cleanup of expired R2 files...")
    try:
        # Initialize r2_storage first to avoid reference before assignment
        r2_storage = CloudflareR2Storage()
        await asyncio.to_thread(r2_storage.cleanup_expired_files)
    except Exception as e:
        print(f"Error during scheduled R2 cleanup: {e}")

# Define the lifespan manager before app initialization
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code
    scheduler = AsyncIOScheduler()
    scheduler.add_job(cleanup_r2_expired_files, 'interval', hours=6)
    scheduler.start()
    print("Scheduler started: R2 cleanup will run every 6 hours")
    
    yield  # This is where the app runs
    
    # Shutdown code
    scheduler.shutdown()
    print("Scheduler shut down")

# Now initialize the app after defining the lifespan function
app = FastAPI(
    title="Enhanced RAG API", 
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://gpt-frontend-five.vercel.app", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
)

# Now initialize r2_storage after app is defined
r2_storage = CloudflareR2Storage()

active_rag_sessions: Dict[str, EnhancedRAG] = {}
sessions_lock = asyncio.Lock()

LOCAL_DATA_BASE_PATH = os.getenv("LOCAL_DATA_PATH", "local_rag_data")
LOCAL_KB_INDEX_PATH_TEMPLATE = os.path.join(LOCAL_DATA_BASE_PATH, "kb_indexes", "{gpt_id}")
LOCAL_USER_INDEX_BASE_PATH = os.path.join(LOCAL_DATA_BASE_PATH, "user_indexes")
TEMP_DOWNLOAD_PATH = os.path.join(LOCAL_DATA_BASE_PATH, "temp_downloads")

os.makedirs(os.path.join(LOCAL_DATA_BASE_PATH, "kb_indexes"), exist_ok=True)
os.makedirs(LOCAL_USER_INDEX_BASE_PATH, exist_ok=True)
os.makedirs(TEMP_DOWNLOAD_PATH, exist_ok=True)

# --- Pydantic Models ---
class BaseRAGRequest(BaseModel):
    user_id: str
    gpt_id: str
    gpt_name: Optional[str] = "default_gpt"

class ChatPayload(BaseModel):
    message: str
    history: Optional[List[Dict[str, str]]] = []
    user_document_keys: Optional[List[str]] = Field([], alias="user_documents")
    use_hybrid_search: Optional[bool] = False
    model: Optional[str] = None
    system_prompt: Optional[str] = None
    web_search_enabled: Optional[bool] = False
    mcp_enabled: Optional[bool] = False
    mcp_schema: Optional[str] = None
    api_keys: Optional[Dict[str, str]] = Field(default_factory=dict)  # Add API keys field

class ChatStreamRequest(BaseRAGRequest, ChatPayload):
    memory: Optional[List[Dict[str, str]]] = []

class ChatRequest(BaseRAGRequest, ChatPayload):
    pass

class GptContextSetupRequest(BaseRAGRequest):
    kb_document_urls: Optional[List[str]] = []
    default_model: Optional[str] = None
    default_system_prompt: Optional[str] = None
    default_use_hybrid_search: Optional[bool] = False
    mcp_enabled_config: Optional[bool] = Field(None, alias="mcpEnabled")
    mcp_schema_config: Optional[str] = Field(None, alias="mcpSchema")

class FileUploadInfoResponse(BaseModel):
    filename: str
    stored_url_or_key: str
    status: str
    error_message: Optional[str] = None

class GptOpenedRequest(BaseModel):
    user_id: str
    gpt_id: str
    gpt_name: str
    file_urls: List[str] = []
    use_hybrid_search: bool = False
    config_schema: Optional[Dict[str, Any]] = Field(default=None, alias="schema")
    api_keys: Optional[Dict[str, str]] = Field(default_factory=dict)

# --- Helper Functions ---
def get_session_id(user_id: str, gpt_id: str) -> str:
    # user_id is expected to be a safe string for session IDs.
    # No replacement needed like for emails.
    return f"user_{user_id}_gpt_{gpt_id}"

async def get_or_create_rag_instance(
    user_id: str,
    gpt_id: str,
    gpt_name: Optional[str] = "default_gpt",
    default_model: Optional[str] = None,
    default_system_prompt: Optional[str] = None,
    default_use_hybrid_search: Optional[bool] = False,
    initial_mcp_enabled_config: Optional[bool] = None,
    initial_mcp_schema_config: Optional[str] = None,
    api_keys: Optional[Dict[str, str]] = None
) -> EnhancedRAG:
    async with sessions_lock:
        if gpt_id in active_rag_sessions:
            rag_instance = active_rag_sessions[gpt_id]
            
            # Update MCP configuration if provided
            if initial_mcp_enabled_config is not None:
                rag_instance.mcp_enabled = initial_mcp_enabled_config
                if initial_mcp_schema_config:
                    try:
                        schema = json.loads(initial_mcp_schema_config)
                        if isinstance(schema, dict) and "mcpServers" in schema:
                            rag_instance.mcp_servers_config = schema["mcpServers"]
                            rag_instance.gpt_mcp_full_schema_str = initial_mcp_schema_config
                            print(f"✅ Updated MCP configuration with servers: {list(rag_instance.mcp_servers_config.keys())}")
                    except json.JSONDecodeError as e:
                        print(f"⚠️ Failed to update MCP schema: {e}")
            
            # Update other configurations...
            if default_model:
                rag_instance.default_llm_model_name = default_model
            if default_system_prompt:
                rag_instance.default_system_prompt = default_system_prompt
            if default_use_hybrid_search is not None:
                rag_instance.default_use_hybrid_search = default_use_hybrid_search
                
            # Update API keys if a RAG instance already exists
            if api_keys:
                # Update OpenAI API key if provided
                if 'openai' in api_keys and api_keys['openai']:
                    old_key = rag_instance.openai_api_key
                    new_key = api_keys['openai']
                    if old_key != new_key:
                        rag_instance.openai_api_key = new_key
                        # Update OpenAI client with new key
                        if hasattr(rag_instance, "async_openai_client"):
                            # Try to reinitialize OpenAI client with new key
                            try:
                                import httpx
                                from openai import AsyncOpenAI
                                timeout_config = httpx.Timeout(connect=15.0, read=180.0, write=15.0, pool=15.0)
                                rag_instance.async_openai_client = AsyncOpenAI(
                                    api_key=new_key,
                                    timeout=timeout_config,
                                    max_retries=1
                                )
                                print(f"✅ OpenAI client reinitialized with user-provided API key")
                            except Exception as e:
                                print(f"❌ Error reinitializing OpenAI client: {e}")
                
                # Update other API keys similarly if they exist in the instance
                for key_name in ['claude', 'gemini', 'groq', 'tavily', 'openrouter']:
                    if key_name in api_keys and api_keys[key_name] and hasattr(rag_instance, f"{key_name}_api_key"):
                        attr_name = f"{key_name}_api_key"
                        if getattr(rag_instance, attr_name) != api_keys[key_name]:
                            setattr(rag_instance, attr_name, api_keys[key_name])
                            print(f"✅ Updated {key_name} API key for RAG instance {gpt_id}")
                
            print(f"Reusing EnhancedRAG instance for gpt_id: {gpt_id}. Updated defaults and API keys if provided.")

        else:
            print(f"Creating new EnhancedRAG instance for gpt_id: {gpt_id}")
            
            # Use API keys from frontend if available, otherwise fallback to environment
            openai_api_key = api_keys.get('openai') if api_keys and 'openai' in api_keys else os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError("OPENAI_API_KEY not set in environment or not provided by frontend.")
                
            qdrant_url = os.getenv("QDRANT_URL")
            qdrant_api_key = os.getenv("QDRANT_API_KEY")
            
            if not qdrant_url:
                raise ValueError("QDRANT_URL not set in environment.")
                
            # Get optional API keys for other providers from frontend or environment
            tavily_api_key = api_keys.get('tavily') if api_keys and 'tavily' in api_keys else os.getenv("TAVILY_API_KEY")
            claude_api_key = api_keys.get('claude') if api_keys and 'claude' in api_keys else os.getenv("ANTHROPIC_API_KEY")
            gemini_api_key = api_keys.get('gemini') if api_keys and 'gemini' in api_keys else os.getenv("GOOGLE_API_KEY")
            groq_api_key = api_keys.get('groq') if api_keys and 'groq' in api_keys else os.getenv("GROQ_API_KEY")
            openrouter_api_key = api_keys.get('openrouter') if api_keys and 'openrouter' in api_keys else os.getenv("OPENROUTER_API_KEY")

            active_rag_sessions[gpt_id] = EnhancedRAG(
                gpt_id=gpt_id,
                r2_storage_client=r2_storage,
                openai_api_key=openai_api_key,
                default_llm_model_name=default_model or os.getenv("DEFAULT_OPENAI_MODEL", "gpt-4o"),
                qdrant_url=qdrant_url,
                qdrant_api_key=qdrant_api_key,
                temp_processing_path=TEMP_DOWNLOAD_PATH,
                default_system_prompt=default_system_prompt,
                default_use_hybrid_search=default_use_hybrid_search,
                tavily_api_key=tavily_api_key,
                initial_mcp_enabled_config=initial_mcp_enabled_config,
                initial_mcp_schema_config=initial_mcp_schema_config
            )
            
            # Update API keys for other providers if available
            rag_instance = active_rag_sessions[gpt_id]
            if claude_api_key and hasattr(rag_instance, "claude_api_key"):
                rag_instance.claude_api_key = claude_api_key
                # Reinitialize Anthropic client if possible
                if hasattr(rag_instance, "anthropic_client") and CLAUDE_AVAILABLE:
                    import anthropic
                    rag_instance.anthropic_client = anthropic.AsyncAnthropic(api_key=claude_api_key)
                    print(f"✅ Claude client reinitialized with user-provided API key")
                
            if gemini_api_key and hasattr(rag_instance, "gemini_api_key"):
                rag_instance.gemini_api_key = gemini_api_key
                # Reinitialize Gemini client if possible
                if hasattr(rag_instance, "gemini_client") and GEMINI_AVAILABLE:
                    import google.generativeai as genai
                    genai.configure(api_key=gemini_api_key)
                    rag_instance.gemini_client = genai
                    print(f"✅ Gemini client reinitialized with user-provided API key")
                
            if groq_api_key and hasattr(rag_instance, "groq_api_key"):
                rag_instance.groq_api_key = groq_api_key
                # Reinitialize Groq client if possible
                if hasattr(rag_instance, "groq_client") and GROQ_AVAILABLE:
                    from groq import AsyncGroq
                    rag_instance.groq_client = AsyncGroq(api_key=groq_api_key)
                    print(f"✅ Groq client reinitialized with user-provided API key")

            if openrouter_api_key and hasattr(rag_instance, "openrouter_api_key"):
                rag_instance.openrouter_api_key = openrouter_api_key
                # Reinitialize OpenRouter client if possible
                if hasattr(rag_instance, "openrouter_client"):
                    try:
                        import httpx
                        from openai import AsyncOpenAI
                        timeout_config = httpx.Timeout(connect=15.0, read=180.0, write=15.0, pool=15.0)
                        rag_instance.openrouter_client = AsyncOpenAI(
                            api_key=openrouter_api_key,
                            base_url="https://openrouter.ai/api/v1",
                            timeout=timeout_config,
                            max_retries=1
                        )
                        print(f"✅ OpenRouter client reinitialized with user-provided API key")
                    except Exception as e:
                        print(f"❌ Error reinitializing OpenRouter client: {e}")

            # Update other API keys similarly if they exist in the instance
            for key_name in ['claude', 'gemini', 'groq', 'tavily', 'openrouter']:
                if key_name in api_keys and api_keys[key_name] and hasattr(rag_instance, f"{key_name}_api_key"):
                    attr_name = f"{key_name}_api_key"
                    if getattr(rag_instance, attr_name) != api_keys[key_name]:
                        setattr(rag_instance, attr_name, api_keys[key_name])
                        print(f"✅ Updated {key_name} API key for RAG instance {gpt_id}")

        return active_rag_sessions[gpt_id]

async def _process_uploaded_file_to_r2(
    file: UploadFile,
    is_user_doc: bool
) -> FileUploadInfoResponse:
    try:
        file_content = await file.read()
        file_bytes_io = BytesIO(file_content)
        
        success, r2_path_or_error = await asyncio.to_thread(
            r2_storage.upload_file,
            file_data=file_bytes_io,
            filename=file.filename,
            is_user_doc=is_user_doc
        )

        if success:
            print(f"File '{file.filename}' (is_user_doc={is_user_doc}) stored at: {r2_path_or_error}")
            return FileUploadInfoResponse(
                filename=file.filename,
                stored_url_or_key=r2_path_or_error,
                status="success"
            )
        else:
            print(f"Failed to store file '{file.filename}'. Error: {r2_path_or_error}")
            return FileUploadInfoResponse(
                filename=file.filename,
                stored_url_or_key="", status="failure", error_message=r2_path_or_error
            )
    except Exception as e:
        print(f"Exception processing file '{file.filename}': {e}")
        return FileUploadInfoResponse(
            filename=file.filename,
            stored_url_or_key="", status="failure", error_message=str(e)
        )

# --- API Endpoints ---

@app.post("/setup-gpt-context", summary="Initialize/update a GPT's knowledge base from URLs and set defaults")
async def setup_gpt_context_endpoint(request: GptContextSetupRequest, background_tasks: BackgroundTasks):
    print(f"Received setup GPT context request for GPT ID: {request.gpt_id}, User: {request.user_id}")
    print(f"KB URLs: {request.kb_document_urls}, Model: {request.default_model}, Prompt: {request.default_system_prompt}, Hybrid: {request.default_use_hybrid_search}")
    print(f"MCP Enabled Config: {request.mcp_enabled_config}, MCP Schema Config Present: {bool(request.mcp_schema_config)}")

    try:
        rag_instance = await get_or_create_rag_instance(
            request.user_id,
            request.gpt_id,
            request.gpt_name,
            default_model=request.default_model,
            default_system_prompt=request.default_system_prompt,
            default_use_hybrid_search=request.default_use_hybrid_search,
            initial_mcp_enabled_config=request.mcp_enabled_config,
            initial_mcp_schema_config=request.mcp_schema_config
        )

        async def _process_kb_urls_task(urls: List[str], rag: EnhancedRAG):
            if urls:
                print(f"Background task: Updating KB for GPT {rag.gpt_id} with {len(urls)} URLs.")
                await rag.update_knowledge_base_from_r2(urls)
                print(f"Background task: KB update finished for GPT {rag.gpt_id}.")
            else:
                print(f"Background task: No URLs provided for KB update for GPT {rag.gpt_id}.")

        if request.kb_document_urls:
            background_tasks.add_task(_process_kb_urls_task, request.kb_document_urls, rag_instance)
            return JSONResponse(content={"success": True, "message": "Knowledge base update initiated in background.", "collection_name": rag_instance.kb_collection_name})
        else:
            # Ensure instance is created/updated even if no URLs are processed immediately
             return JSONResponse(content={"success": True, "message": "GPT context (defaults, MCP config) initialized/updated. No KB URLs to process.", "collection_name": rag_instance.kb_collection_name if rag_instance.kb_retriever else None})

    except Exception as e:
        print(f"Error in setup_gpt_context_endpoint for GPT {request.gpt_id}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to setup GPT context: {str(e)}")

@app.post("/upload-documents", summary="Upload documents (KB or User-specific) including images")
async def upload_documents_endpoint(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    user_id: str = Form(...),
    gpt_id: str = Form(...),
    is_user_document: str = Form("false"),
):
    is_user_doc_bool = is_user_document.lower() == "true"
    processing_results: List[FileUploadInfoResponse] = []
    r2_keys_or_urls_for_indexing: List[str] = []

    for file_upload in files:
        # Check if it's an image file
        filename = file_upload.filename
        is_image = False
        if filename:
            _, ext = os.path.splitext(filename)
            ext = ext.lower()
            is_image = ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
            
        # Log file type
        if is_image:
            print(f"Processing image file: {filename}")
        
        result = await _process_uploaded_file_to_r2(file_upload, is_user_doc_bool)
        processing_results.append(result)
        if result.status == "success" and result.stored_url_or_key:
            r2_keys_or_urls_for_indexing.append(result.stored_url_or_key)

    if not r2_keys_or_urls_for_indexing:
        return JSONResponse(status_code=400, content={
            "message": "No files were successfully uploaded to R2.",
            "upload_results": [r.model_dump() for r in processing_results]
        })

    rag_instance = await get_or_create_rag_instance(user_id=user_id, gpt_id=gpt_id)
    
    async def _index_documents_task(rag: EnhancedRAG, keys_or_urls: List[str], is_user_specific: bool, u_id: str, g_id: str):
        doc_type = "user-specific" if is_user_specific else "knowledge base"
        s_id = get_session_id(u_id, g_id)
        print(f"BG Task: Indexing {len(keys_or_urls)} {doc_type} documents for gpt_id '{g_id}' (session '{s_id}')...")
        try:
            if is_user_specific:
                await rag.update_user_documents_from_r2(session_id=s_id, r2_keys_or_urls=keys_or_urls)
            else:
                await rag.update_knowledge_base_from_r2(keys_or_urls)
            print(f"BG Task: Indexing complete for {doc_type} documents.")
        except Exception as e:
            print(f"BG Task: Error indexing {doc_type} documents for gpt_id '{g_id}': {e}")

    background_tasks.add_task(_index_documents_task, rag_instance, r2_keys_or_urls_for_indexing, is_user_doc_bool, user_id, gpt_id)

    return JSONResponse(status_code=202, content={
        "message": f"{len(r2_keys_or_urls_for_indexing)} files accepted for {'user-specific' if is_user_doc_bool else 'knowledge base'} indexing. Processing in background.",
        "upload_results": [r.model_dump() for r in processing_results]
    })

@app.post("/chat-stream")
async def chat_stream(request: ChatStreamRequest):
    try:
        session_id = get_session_id(request.user_id, request.gpt_id)
        
        print(f"Chat stream request from {request.user_id} for GPT {request.gpt_id}")
        print(f"MCP enabled: {request.mcp_enabled}")
        print(f"Web search enabled: {request.web_search_enabled}")
        
        # Determine if this is a new chat
        is_new_chat = not request.history and not request.memory
        
        # Parse and validate MCP configuration
        mcp_schema_str = None
        if request.mcp_enabled and request.mcp_schema:
            try:
                # Validate schema structure
                mcp_config = json.loads(request.mcp_schema)
                if "mcpServers" in mcp_config and isinstance(mcp_config["mcpServers"], dict):
                    mcp_schema_str = request.mcp_schema
                    print(f"✅ Valid MCP schema received with servers: {list(mcp_config['mcpServers'].keys())}")
                else:
                    print("⚠️ Invalid MCP schema structure")
            except json.JSONDecodeError as e:
                print(f"⚠️ Error parsing MCP schema: {e}")
        
        rag = await get_or_create_rag_instance(
            user_id=request.user_id,
            gpt_id=request.gpt_id,
            gpt_name=request.gpt_name,
            initial_mcp_enabled_config=request.mcp_enabled,
            initial_mcp_schema_config=mcp_schema_str,
            api_keys=request.api_keys
        )
        
        async def generate():
            async for chunk in rag.query_stream(
                session_id=session_id,
                query=request.message,
                chat_history=request.history,
                user_r2_document_keys=request.user_document_keys,
                use_hybrid_search=True,  # Always enable hybrid search
                llm_model_name=request.model,
                system_prompt_override=request.system_prompt,
                enable_web_search=request.web_search_enabled,
                mcp_enabled=request.mcp_enabled,
                mcp_schema=request.mcp_schema,
                api_keys=request.api_keys,
                is_new_chat=is_new_chat  # Pass the new parameter
            ):
                if chunk["type"] == "error":
                    yield f"data: {json.dumps({'error': chunk.get('text', chunk.get('error', 'Unknown error'))})}\n\n"
                else:
                    yield f"data: {json.dumps(chunk)}\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream"
        )
                    
    except Exception as e:
        error_message = f"Error in chat stream: {str(e)}"
        print(error_message)
        return StreamingResponse(
            (f"data: {json.dumps({'error': error_message})}\n\n" for _ in range(1)),
            media_type="text/event-stream"
        )

@app.post("/chat", summary="Handle non-streaming chat requests")
async def chat_endpoint(request: ChatRequest):
    try:
        session_id = get_session_id(request.user_id, request.gpt_id)
        
        rag = await get_or_create_rag_instance(
            user_id=request.user_id,
            gpt_id=request.gpt_id,
            gpt_name=request.gpt_name,
            api_keys=request.api_keys
        )
        
        # Get full response using the existing query method, which will
        # now utilize the auto-detection if mcp_enabled is None
        response = await rag.query(
            session_id=session_id,
            query=request.message,
            chat_history=request.history,
            user_r2_document_keys=request.user_document_keys,
            use_hybrid_search=request.use_hybrid_search,
            llm_model_name=request.model,
            system_prompt_override=request.system_prompt,
            enable_web_search=request.web_search_enabled
        )
        
        return response
        
    except Exception as e:
        error_message = f"Error in chat endpoint: {str(e)}"
        print(error_message)
        return JSONResponse(
            status_code=500,
            content={"error": error_message}
        )

@app.post("/gpt-opened", summary="Notify backend when a GPT is opened, ensure context is set up.")
async def gpt_opened_endpoint(request: GptOpenedRequest, background_tasks: BackgroundTasks):
    session_id = get_session_id(request.user_id, request.gpt_id)
    print(f"GPT opened: ID={request.gpt_id}, Name='{request.gpt_name}', User={request.user_id}")

    try:
        # Extract MCP configuration
        mcp_enabled = False
        mcp_schema_str = None
        
        if request.config_schema:
            # Get MCP configuration
            mcp_enabled = request.config_schema.get("mcpEnabled", False)
            mcp_schema = request.config_schema.get("mcpSchema")
            
            if isinstance(mcp_schema, str):
                try:
                    # Validate the schema is proper JSON
                    json.loads(mcp_schema)
                    mcp_schema_str = mcp_schema
                except json.JSONDecodeError:
                    print(f"Warning: Invalid MCP schema JSON string provided")
                    mcp_schema_str = None
            elif isinstance(mcp_schema, dict):
                try:
                    mcp_schema_str = json.dumps(mcp_schema)
                except Exception as e:
                    print(f"Warning: Could not serialize MCP schema dict to JSON: {e}")
                    mcp_schema_str = None

            print(f"MCP Configuration: enabled={mcp_enabled}, schema_valid={bool(mcp_schema_str)}")
            if mcp_schema_str:
                print(f"Available MCP servers: {list(json.loads(mcp_schema_str).get('mcpServers', {}).keys())}")

        rag_instance = await get_or_create_rag_instance(
            user_id=request.user_id,
            gpt_id=request.gpt_id,
            gpt_name=request.gpt_name,
            default_model=request.config_schema.get("model") if request.config_schema else None,
            default_system_prompt=request.config_schema.get("instructions") if request.config_schema else None,
            default_use_hybrid_search=request.use_hybrid_search,
            initial_mcp_enabled_config=mcp_enabled,
            initial_mcp_schema_config=mcp_schema_str,
            api_keys=request.api_keys
        )

        async def _process_kb_urls_task(urls: List[str], rag: EnhancedRAG):
            if urls:
                print(f"Background task (gpt-opened): Updating KB for GPT {rag.gpt_id} with {len(urls)} URLs.")
                await rag.update_knowledge_base_from_r2(urls)
                print(f"Background task (gpt-opened): KB update finished for GPT {rag.gpt_id}.")
            else:
                print(f"Background task (gpt-opened): No KB URLs to process for GPT {rag.gpt_id}.")

        if request.file_urls: # These are KB files for the GPT
            background_tasks.add_task(_process_kb_urls_task, request.file_urls, rag_instance)
        
        return JSONResponse(content={
            "success": True,
            "message": f"GPT '{request.gpt_name}' context initialized/updated for user {request.user_id}.",
            "collection_name": rag_instance.kb_collection_name,
            "session_id": session_id,
            "mcp_config_loaded": {
                "enabled": rag_instance.gpt_mcp_enabled_config,
                "schema_present": bool(rag_instance.gpt_mcp_full_schema_str)
            }
        })
    except Exception as e:
        print(f"Error in gpt_opened_endpoint for GPT {request.gpt_id}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to handle GPT opened event: {str(e)}")

@app.post("/upload-chat-files", summary="Upload files for chat including images")
async def upload_chat_files_endpoint(
    files: List[UploadFile] = File(...),
    user_id: str = Form(...),
    gpt_id: str = Form(...),
    gpt_name: str = Form(...),
    collection_name: str = Form(...),
    is_user_document: str = Form("true"),
    use_hybrid_search: str = Form("false"),
    optimize_pdfs: str = Form("false"),
):
    is_user_doc_bool = is_user_document.lower() == "true"
    use_hybrid_search_bool = use_hybrid_search.lower() == "true"
    optimize_pdfs_bool = optimize_pdfs.lower() == "true"
    
    processing_results = []
    file_urls = []

    for file_upload in files:
        # Check if it's an image file
        filename = file_upload.filename
        is_image = False
        if filename:
            _, ext = os.path.splitext(filename)
            ext = ext.lower()
            is_image = ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
            
        # Log file type
        if is_image:
            print(f"Processing image file for chat: {filename}")
            
        result = await _process_uploaded_file_to_r2(file_upload, is_user_doc_bool)
        if result.status == "success" and result.stored_url_or_key:
            file_urls.append(result.stored_url_or_key)
        processing_results.append(result)

    rag_instance = await get_or_create_rag_instance(
        user_id=user_id, 
        gpt_id=gpt_id,
        gpt_name=gpt_name
    )
    
    if file_urls:
        session_id = get_session_id(user_id, gpt_id)
        
        try:
            if is_user_doc_bool:
                await rag_instance.update_user_documents_from_r2(session_id=session_id, r2_keys_or_urls=file_urls)
            else:
                await rag_instance.update_knowledge_base_from_r2(file_urls)
            print(f"Indexing complete for {len(file_urls)} {'user-specific' if is_user_doc_bool else 'knowledge base'} documents for session '{session_id}'.")
        except Exception as e:
            print(f"Error indexing chat files for session '{session_id}': {e}")
            return {
                "success": False,
                "message": f"Failed to index {len(file_urls)} files: {str(e)}",
                "file_urls": file_urls,
                "processing": False
            }
    
    return {
        "success": True,
        "message": f"Processed and indexed {len(file_urls)} files",
        "file_urls": file_urls,
        "processing": len(file_urls) > 0
    }

@app.get("/gpt-collection-info/{param1}/{param2}", summary="Get information about a GPT collection")
async def gpt_collection_info(param1: str, param2: str):
    return {
        "status": "available",
        "timestamp": time.time()
    }

@app.get("/", include_in_schema=False)
async def root_redirect():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")

@app.get("/health", summary="Health check endpoint", tags=["Monitoring"])
async def health_check():
    return {"status": "healthy", "timestamp": time.time()}

@app.post("/dev/reset-gpt-context", summary="DEVELOPMENT ONLY: Clear RAG context for a GPT", tags=["Development"])
async def dev_reset_gpt_context_endpoint(gpt_id: str = Form(...)):
    if os.getenv("ENVIRONMENT_TYPE", "production").lower() != "development":
        return JSONResponse(status_code=403, content={"error": "Endpoint only available in development."})

    async with sessions_lock:
        if gpt_id in active_rag_sessions:
            try:
                rag_instance_to_reset = active_rag_sessions.pop(gpt_id)
                await rag_instance_to_reset.clear_all_context()
                
                kb_index_path_to_delete = LOCAL_KB_INDEX_PATH_TEMPLATE.format(gpt_id=gpt_id)
                if os.path.exists(kb_index_path_to_delete):
                    shutil.rmtree(kb_index_path_to_delete)
                
                print(f"DEV: Cleared in-memory RAG context and local KB index for gpt_id '{gpt_id}'. R2 files not deleted.")
                return {"status": "success", "message": f"RAG context for gpt_id '{gpt_id}' cleared from memory and local disk."}
            except Exception as e:
                print(f"DEV: Error clearing context for gpt_id '{gpt_id}': {e}")
                return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
        else:
            return JSONResponse(status_code=404, content={"status": "not_found", "message": f"No active RAG context for gpt_id '{gpt_id}'."})

@app.post("/maintenance/cleanup-r2", summary="Manually trigger cleanup of expired R2 files", tags=["Maintenance"])
async def manual_cleanup_r2():
    try:
        await asyncio.to_thread(r2_storage.cleanup_expired_files)
        return {"status": "success", "message": "R2 cleanup completed"}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Error during R2 cleanup: {str(e)}"}
        )

@app.post("/index-knowledge")
async def index_knowledge_endpoint(
    request: Request,
    background_tasks: BackgroundTasks
):
    try:
        # Parse request body
        data = await request.json()
        gpt_id = data.get("gpt_id")
        file_urls = data.get("file_urls", [])
        user_id = data.get("user_id", "user_example_id")
        system_prompt = data.get("system_prompt", "")
        use_hybrid_search = data.get("use_hybrid_search", False)
        schema = data.get("schema", {})
        
        if not gpt_id:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "Missing gpt_id parameter"}
            )
        
        # Initialize RAG instance
        rag_instance = await get_or_create_rag_instance(
            user_id=user_id,
            gpt_id=gpt_id,
            gpt_name=schema.get("name", "Custom GPT"),
            default_model=schema.get("model"),
            default_system_prompt=system_prompt,
            default_use_hybrid_search=use_hybrid_search
        )
        
        # Process files in background
        async def _process_kb_urls_task(urls: List[str], rag: EnhancedRAG):
            print(f"BG Task: Processing {len(urls)} KB URLs for gpt_id '{rag.gpt_id}'...")
            try:
                await rag.update_knowledge_base_from_r2(urls)
                print(f"BG Task: Knowledge base update from URLs complete for gpt_id '{rag.gpt_id}'.")
            except Exception as e:
                print(f"Error during background KB update for gpt_id '{rag.gpt_id}': {e}")

        # Start background task for file processing
        if file_urls:
            background_tasks.add_task(_process_kb_urls_task, file_urls, rag_instance)
        
        return {"success": True, "message": f"Indexing initiated for {len(file_urls)} files."}
    
    except Exception as e:
        print(f"Error in index-knowledge endpoint: {e}")
        return JSONResponse(
            status_code=500, 
            content={"success": False, "error": str(e)}
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)