import os
import re
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import requests
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from groq import Groq

# Ensure local backend modules are importable whether running from project root
# (`uvicorn Backend.main:app`) or from inside `Backend/`.
BACKEND_DIR = os.path.dirname(__file__)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from intent import IntentAnalyzer  # noqa: E402
from pydantic import BaseModel  # noqa: E402
from reviews import get_review_summary, get_reviews, get_user_review, submit_review  # noqa: E402
from search import CACHE_VERSION, ModelSearchEngine  # noqa: E402
from tripo_adapter import generate_from_image  # noqa: E402

# Simple in-memory job store for async generation jobs
generation_jobs: dict = {}
# Executor for running generation tasks in background threads
generation_executor = ThreadPoolExecutor(max_workers=2)

# Load environment variables from `Backend/.env`
load_dotenv()

groq_api_key = os.environ.get("GROQ_API_KEY")
# If no API key is configured, keep `client` as None and handle gracefully in endpoints.
client = Groq(api_key=groq_api_key) if groq_api_key else None

# Free AI provider fallback (OpenRouter/OpenAI/Anthropic-compatible)
FREE_AI_API_PROVIDER = os.getenv("FREE_AI_API_PROVIDER")
FREE_AI_API_KEY = os.getenv("FREE_AI_API_KEY")
FREE_AI_API_URL = os.getenv("FREE_AI_API_URL")
FREE_AI_API_MODEL = os.getenv("FREE_AI_API_MODEL")

app = FastAPI(title="3D Model Generation API")

# Configure allowed origins from env (comma-separated). Defaults to open for local/dev convenience.
raw_origins = (os.getenv("ALLOWED_ORIGINS") or "*").strip()
allowed_origins = [o.strip() for o in raw_origins.split(",") if o.strip()] or ["*"]

# Setup CORS for the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

models_dir = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(models_dir, exist_ok=True)
app.mount("/models", StaticFiles(directory=models_dir), name="models")

intent_analyzer = IntentAnalyzer()
search_engine = ModelSearchEngine()


class QueryRequest(BaseModel):
    query: str


@app.post("/api/intent")
async def analyze_intent(request: QueryRequest):
    """
    Expands the user prompt into primary keywords, structural components, and context.
    Provides fallback capabilities using Gemini if configured.
    """
    try:
        result = intent_analyzer.parse(request.query)
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/search")
async def search_models(request: QueryRequest):
    """
    Takes the parsed intent and queries external APIs to find the best 3D models.
    """
    try:
        # First, analyze intent (or expect the frontend to pass the intent)
        intent = intent_analyzer.parse(request.query)

        # Search using the intent keywords
        results = search_engine.search(intent)

        if not results:
            # Fallback 1: Procedural generation metadata
            return {
                "status": "fallback",
                "message": "No specific models found, using procedural generation.",
                "data": {
                    "type": "procedural",
                    "components": intent.get("structural_components", ["sphere", "box"]),
                },
            }

        return {"status": "success", "data": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ChatRequest(BaseModel):
    message: str
    model_context: Optional[str] = None


class AgentQuestionRequest(BaseModel):
    concept: str
    question: str
    model_name: Optional[str] = None


class CacheClearRequest(BaseModel):
    query: Optional[str] = None


class ReviewRequest(BaseModel):
    model_id: str
    user_id: str
    rating: int
    comment: Optional[str] = ""


class LabelPositioningRequest(BaseModel):
    model_id: str
    concept: str
    part_definitions: list
    model_image_base64: str  # Base64-encoded PNG/JPG image


def _clean_agent_answer_text(text: str) -> str:
    if not text:
        return ""

    cleaned = text.strip()
    lowered = cleaned.lower()

    leadins = [
        "based on the provided wikipedia context,",
        "based on the provided context,",
        "based on the context,",
        "from the provided wikipedia context,",
        "from the provided context,",
    ]
    for lead in leadins:
        if lowered.startswith(lead):
            cleaned = cleaned[len(lead) :].strip(" ,:")
            lowered = cleaned.lower()

    replacements = {
        "provided wikipedia context": "available information",
        "wikipedia context": "available information",
        "provided context": "available information",
    }
    for old, new in replacements.items():
        cleaned = re.sub(old, new, cleaned, flags=re.IGNORECASE)

    if not cleaned:
        cleaned = "I don't have enough information to answer that accurately right now."

    if not cleaned.endswith((".", "!", "?")):
        cleaned += "."

    return cleaned


def _extract_topic_from_question(question: str) -> Optional[str]:
    q = (question or "").strip().lower()
    if not q:
        return None

    # Handle short prompts like "table?" or "chair".
    simple = re.sub(r"[^a-z0-9\-\s]", "", q).strip()
    if simple and len(simple.split()) <= 3:
        return simple

    patterns = [
        r"^(?:what\s+is|what\s+are|define|meaning\s+of)\s+(?:an?\s+|the\s+)?([a-z0-9\-\s]{2,80})\??$",
        r"^(?:tell\s+me\s+about|explain)\s+(?:an?\s+|the\s+)?([a-z0-9\-\s]{2,80})\??$",
    ]
    for pattern in patterns:
        match = re.match(pattern, q)
        if not match:
            continue
        topic = re.sub(r"\s+", " ", match.group(1)).strip(" ?.,!:")
        if topic:
            return topic

    return None


def _local_topic_fallback(topic: str) -> Optional[str]:
    if not topic:
        return None

    topic_key = topic.strip().lower()
    glossary = {
        "table": "A table is a piece of furniture with a flat top supported by legs, used for activities like eating, working, or placing objects.",
        "chair": "A chair is a seat with a backrest designed for one person, often supported by four legs.",
        "sofa": "A sofa is a padded multi-seat furniture item designed for sitting and lounging.",
        "lamp": "A lamp is a device that emits light, usually using an electric bulb and a supporting base or stand.",
        "desk": "A desk is a work table used for reading, writing, or computer work, often with drawers for storage.",
    }

    return glossary.get(topic_key)


def _get_wikipedia_fallback(concept: str, question: str) -> str:
    question_topic = _extract_topic_from_question(question)
    normalized_concept = (concept or "").strip()
    generic_contexts = {"", "3d model viewing", "3d model", "model viewing", "unknown", "this concept"}
    concept_is_generic = normalized_concept.lower() in generic_contexts
    safe_concept = (question_topic if concept_is_generic else normalized_concept) or (question_topic or normalized_concept or "3D design")
    safe_concept = safe_concept.strip() or "3D design"

    # In this app context, user questions are about the currently viewed 3D object.
    # If question topic matches a known physical-object term, prefer object meaning.
    if question_topic:
        local_answer = _local_topic_fallback(question_topic)
        if local_answer:
            if concept_is_generic:
                return local_answer
            if normalized_concept and question_topic in normalized_concept.lower():
                return local_answer

    try:
        # Local import keeps backend startup resilient if wikipedia dependency is unavailable.
        from wikipedia_api import get_wikipedia_summary

        summary = (get_wikipedia_summary(safe_concept, max_sentences=4) or "").strip()
    except Exception as e:
        print(f"wikipedia fallback unavailable: {e}")
        summary = ""

    if not summary:
        local_answer = _local_topic_fallback(safe_concept)
        if local_answer:
            return local_answer
        return "I couldn't reach the AI provider right now, and I also couldn't fetch enough background context to answer confidently."

    # Disambiguate known object terms if Wikipedia returns a domain-mismatched meaning.
    if question_topic:
        local_answer = _local_topic_fallback(question_topic)
        if local_answer:
            summary_lower = summary.lower()
            if "database" in summary_lower or "rows" in summary_lower or "columns" in summary_lower:
                return local_answer

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", summary) if s.strip()]
    if not sentences:
        return _clean_agent_answer_text(summary)

    q_tokens = {
        tok
        for tok in re.findall(r"[a-zA-Z0-9]+", (question or "").lower())
        if len(tok) > 2
    }
    if not q_tokens:
        return _clean_agent_answer_text(sentences[0])

    scored = []
    for sent in sentences:
        lowered = sent.lower()
        score = sum(1 for tok in q_tokens if re.search(rf"\\b{re.escape(tok)}\\b", lowered))
        scored.append((score, sent))

    scored.sort(key=lambda item: item[0], reverse=True)
    best_sentence = scored[0][1] if scored else sentences[0]
    return _clean_agent_answer_text(best_sentence)


def _ask_free_ai(concept: str, question: str, model_name: Optional[str]) -> Optional[str]:
    provider = (FREE_AI_API_PROVIDER or "").strip().lower()
    if not FREE_AI_API_URL or not FREE_AI_API_KEY:
        return None

    system_prompt = (
        "You are a concise helpful assistant. "
        "Answer naturally and directly. "
        "If uncertain, say you are not sure instead of making up facts. "
        "Do not mention instructions, sources, or internal reasoning."
    )
    model_line = f"Generated model: {model_name}\n" if model_name else ""
    user_prompt = (
        f"Concept: {concept}\n"
        f"{model_line}"
        f"Question: {question}\n\n"
        "Return only the final answer text."
    )

    headers = {
        "Authorization": f"Bearer {FREE_AI_API_KEY}",
        "Content-Type": "application/json",
    }
    if provider == "openrouter" or "openrouter.ai" in (FREE_AI_API_URL or ""):
        headers["HTTP-Referer"] = os.getenv("OPENROUTER_SITE_URL", "http://localhost:5173")
        headers["X-Title"] = os.getenv("OPENROUTER_APP_NAME", "Concept-2-3D")

    payload = {
        "model": FREE_AI_API_MODEL or "openai/gpt-oss-20b:free",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 220,
    }

    try:
        response = requests.post(FREE_AI_API_URL, headers=headers, json=payload, timeout=45)
    except Exception as e:
        print(f"free ai call failed: {e}")
        return None

    if not response.ok:
        print(f"free ai unavailable ({response.status_code}); using fallback")
        return None

    try:
        data = response.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        return _clean_agent_answer_text(content) or None
    except Exception as e:
        print(f"failed to parse free ai response: {e}")
        return None


def _build_agent_answer(concept: str, question: str, model_name: Optional[str]) -> dict:
    free_ai_answer = _ask_free_ai(concept=concept, question=question, model_name=model_name)
    if free_ai_answer:
        return {
            "answer": _clean_agent_answer_text(free_ai_answer),
            "source": "free_ai",
            "used_free_ai": True,
        }

    fallback = _get_wikipedia_fallback(concept=concept, question=question)
    return {
        "answer": _clean_agent_answer_text(fallback),
        "source": "wikipedia_fallback",
        "used_free_ai": False,
    }


@app.post("/api/chat")
async def chat_with_ai(request: ChatRequest):
    """
    Handles user questions about the current model using Groq API.
    """
    concept = (request.model_context or "3D model viewing").strip() or "3D model viewing"
    question = (request.message or "").strip()
    if not question:
        return {"status": "success", "message": "Please ask a question for the AI assistant."}

    provider = (FREE_AI_API_PROVIDER or "").strip().lower()
    try_free_first = provider in {"openrouter", "free_ai", "openai-compatible"}

    if try_free_first:
        agent_resp = _build_agent_answer(concept=concept, question=question, model_name=None)
        return {
            "status": "success",
            "message": agent_resp["answer"],
            "source": agent_resp["source"],
            "used_free_ai": agent_resp["used_free_ai"],
        }

    # Prefer GROQ if configured and FREE_AI wasn't selected first.
    if client is not None:
        try:
            completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": f"You are a helpful AI 3D design assistant. The user is currently viewing a 3D model: {request.model_context or 'Unknown'}. Answer their questions concisely.",
                    },
                    {
                        "role": "user",
                        "content": request.message,
                    },
                ],
                model="llama-3.3-70b-versatile",
            )
            return {
                "status": "success",
                "message": _clean_agent_answer_text(completion.choices[0].message.content),
                "source": "groq",
                "used_free_ai": False,
            }
        except Exception as e:
            print(f"chat completion (groq) error: {e}")

    # Legacy-compatible fallback path: use FREE_AI if available, otherwise curated Wikipedia fallback.
    agent_resp = _build_agent_answer(concept=concept, question=question, model_name=None)
    return {
        "status": "success",
        "message": agent_resp["answer"],
        "source": agent_resp["source"],
        "used_free_ai": agent_resp["used_free_ai"],
    }


@app.post("/api/agent/ask")
def ask_agent_api(payload: AgentQuestionRequest):
    concept = (payload.concept or "").strip()
    question = (payload.question or "").strip()
    model_name = (payload.model_name or "").strip() or None

    if not concept:
        return {"answer": "Please provide a concept first.", "source": "agent", "used_free_ai": False}
    if not question:
        return {"answer": "Please ask a question for the AI agent.", "source": "agent", "used_free_ai": False}

    response = _build_agent_answer(concept=concept, question=question, model_name=model_name)
    return {
        "answer": response["answer"],
        "source": response["source"],
        "used_free_ai": response["used_free_ai"],
        "concept": concept,
        "model_name": model_name,
    }


@app.post("/agent/ask")
def ask_agent_legacy(payload: AgentQuestionRequest):
    # Backward-compatible route expected by older frontend builds.
    return ask_agent_api(payload)


@app.post("/api/cache/clear")
async def clear_cache(request: CacheClearRequest):
    """
    Clears cached search entries.
    - If query is provided, clears only that normalized query key.
    - If omitted, clears all cached entries.
    """
    try:
        if request.query:
            normalized = search_engine._normalize_query(request.query)
            cache_key = f"{CACHE_VERSION}::{normalized}"
            deleted = search_engine.cache.clear_cache(cache_key)
            return {
                "status": "success",
                "cleared": deleted,
                "scope": "single",
                "query": normalized,
            }

        deleted = search_engine.cache.clear_cache()
        return {"status": "success", "cleared": deleted, "scope": "all"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate_from_image")
async def generate_from_image_endpoint(file: UploadFile = File(...)):
    """Generate a 3D model from an uploaded image using TripoSR (if configured).

    The endpoint expects a file upload (png/jpg). If the backend env var
    `TRIPO_COMMAND` is not set, this returns a 501 with instructions.
    """
    try:
        contents = await file.read()
        # persist the original upload for traceability
        uploads_dir = os.path.join(models_dir, "uploads")
        os.makedirs(uploads_dir, exist_ok=True)
        upload_name = f"upload_{uuid.uuid4().hex}.png"
        upload_path = os.path.join(uploads_dir, upload_name)
        with open(upload_path, "wb") as uf:
            uf.write(contents)
        print(f"Saved upload: {upload_path} ({len(contents)} bytes)")
        # Output to models dir so the frontend can request via /models/
        output_dir = models_dir

        # Allow configuration via env vars
        filename_prefix = os.getenv("TRIPO_PREFIX", "tripo")
        timeout = int(os.getenv("TRIPO_TIMEOUT", "600"))

        glb_path = generate_from_image(contents, output_dir, filename_prefix=filename_prefix, timeout=timeout)
        if not glb_path:
            raise HTTPException(
                status_code=501,
                detail=("TripoSR is not configured or generation failed. " "Set the TRIPO_COMMAND environment variable to a command that accepts " "{input_image} and {output_dir} placeholders and produces a .glb file in the output dir."),
            )

        # Build model URL relative to /models/ and preserve subdirectory structure
        rel_path = os.path.relpath(glb_path, models_dir).replace("\\", "/")
        model_url = f"/models/{rel_path}"
        uid = os.path.basename(glb_path)
        return {
            "status": "success",
            "model_url": model_url,
            "uid": uid,
            "upload_path": f"/models/uploads/{upload_name}",
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"generate_from_image error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _run_generation_job(job_id: str, file_bytes: bytes):
    generation_jobs[job_id]["status"] = "running"
    try:
        output_dir = models_dir
        # Save uploaded bytes for traceability in async jobs too
        uploads_dir = os.path.join(models_dir, "uploads")
        os.makedirs(uploads_dir, exist_ok=True)
        upload_name = f"upload_{uuid.uuid4().hex}.png"
        upload_path = os.path.join(uploads_dir, upload_name)
        with open(upload_path, "wb") as uf:
            uf.write(file_bytes)
        generation_jobs[job_id]["upload_path"] = f"/models/uploads/{upload_name}"

        glb_path = generate_from_image(file_bytes, output_dir)
        if not glb_path:
            detailed_error = _latest_tripo_error(output_dir)
            generation_jobs[job_id]["status"] = "failed"
            generation_jobs[job_id]["error"] = detailed_error or "Generation failed or TripoSR not configured"
            return
        rel_path = os.path.relpath(glb_path, models_dir).replace("\\", "/")
        model_url = f"/models/{rel_path}"
        generation_jobs[job_id]["status"] = "done"
        generation_jobs[job_id]["model_url"] = model_url
        generation_jobs[job_id]["uid"] = os.path.basename(glb_path)
    except Exception as e:
        generation_jobs[job_id]["status"] = "failed"
        generation_jobs[job_id]["error"] = str(e)


def _latest_tripo_error(output_dir: str) -> Optional[str]:
    """Return a concise error extracted from latest Tripo run.log, if available."""
    try:
        run_dirs = [
            os.path.join(output_dir, name)
            for name in os.listdir(output_dir)
            if name.startswith("tripo_") and os.path.isdir(os.path.join(output_dir, name))
        ]
        if not run_dirs:
            return None

        run_dirs.sort(key=os.path.getmtime, reverse=True)
        run_log = os.path.join(run_dirs[0], "run.log")
        if not os.path.exists(run_log):
            return None

        with open(run_log, "r", encoding="utf-8", errors="ignore") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]

        if not lines:
            return None

        # Prefer traceback tail when present.
        for ln in reversed(lines):
            if "Error" in ln or "Exception" in ln or "Traceback" in ln or "No such file" in ln:
                return ln

        # Fallback to the last non-empty line.
        return lines[-1]
    except Exception:
        return None


@app.post("/api/generate_from_image_async")
async def generate_from_image_async(file: UploadFile = File(...)):
    """Start a background generation job and return a job id for polling."""
    try:
        contents = await file.read()
        job_id = uuid.uuid4().hex
        generation_jobs[job_id] = {
            "status": "pending",
            "model_url": None,
            "error": None,
        }

        # Submit to executor
        generation_executor.submit(_run_generation_job, job_id, contents)

        return {"status": "accepted", "job_id": job_id}
    except Exception as e:
        print(f"generate_from_image_async error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/generate_status/{job_id}")
async def generate_status(job_id: str):
    job = generation_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": job_id,
        "status": job["status"],
        "model_url": job.get("model_url"),
        "error": job.get("error"),
    }


@app.post("/api/reviews/submit")
async def submit_model_review(request: ReviewRequest):
    """
    Submit or update a review for a model.
    Allows users to rate (1-5 stars) and add optional comments.
    """
    try:
        review = submit_review(
            model_id=request.model_id,
            user_id=request.user_id,
            rating=request.rating,
            comment=request.comment or "",
        )
        return {"status": "success", "review": review}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/reviews/{model_id}")
async def get_model_reviews(model_id: str, limit: int = 50):
    """
    Fetch all reviews for a specific model.
    Returns reviews sorted by newest first.
    """
    try:
        reviews = get_reviews(model_id, limit=limit)
        summary = get_review_summary(model_id)
        return {
            "status": "success",
            "model_id": model_id,
            "summary": summary,
            "reviews": reviews,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/reviews/{model_id}/summary")
async def get_model_review_summary(model_id: str):
    """
    Get aggregate review statistics for a model.
    Returns average rating, total count, and distribution by star rating.
    """
    try:
        summary = get_review_summary(model_id)
        return {"status": "success", "data": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/reviews/{model_id}/user/{user_id}")
async def get_user_model_review(model_id: str, user_id: str):
    """
    Get the current user's review for a specific model.
    Used to pre-populate review form if user has already reviewed.
    """
    try:
        review = get_user_review(model_id, user_id)
        if not review:
            return {"status": "success", "review": None}
        return {"status": "success", "review": review}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/labels/position-from-image")
async def position_labels_from_image(request: LabelPositioningRequest):
    """
    Use Gemini vision API to analyze a model image and generate precise x,y,z coordinates
    for each part label based on visual analysis of the 3D model.

    The frontend captures a screenshot of the 3D model and sends it here along with part definitions.
    Gemini analyzes the image and returns optimized x,y,z coordinates for each label.
    """
    try:
        # Validate input
        if not request.model_image_base64:
            raise ValueError("Model image is required for vision-based positioning")

        if not request.part_definitions or len(request.part_definitions) == 0:
            return {"status": "success", "updated_parts": []}

        # Use Gemini vision to refine positions
        updated_parts = search_engine._get_gemini_label_positions(
            normalized_keywords=request.concept or "model",
            part_definitions=request.part_definitions,
            model_image_base64=request.model_image_base64,
        )

        return {
            "status": "success",
            "model_id": request.model_id,
            "concept": request.concept,
            "updated_parts": updated_parts,
            "message": "Label positions optimized using Gemini vision analysis",
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Label positioning error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
