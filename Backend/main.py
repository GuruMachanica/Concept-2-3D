import os
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
from intent import IntentAnalyzer
from pydantic import BaseModel
from reviews import get_review_summary, get_reviews, get_user_review, submit_review
from search import CACHE_VERSION, ModelSearchEngine
from tripo_adapter import generate_from_image

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


@app.post("/api/chat")
async def chat_with_ai(request: ChatRequest):
    """
    Handles user questions about the current model using Groq API.
    """
    # Prefer GROQ if configured
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
                "message": completion.choices[0].message.content,
            }
        except Exception as e:
            print(f"chat completion (groq) error: {e}")
            raise HTTPException(
                status_code=502,
                detail="Error from Groq chat API: see server logs for details",
            )

    # Otherwise attempt to use FREE_AI_API_URL (OpenRouter/OpenAI-compatible)
    if FREE_AI_API_URL and FREE_AI_API_KEY:
        payload = {
            "model": FREE_AI_API_MODEL or "gpt-4o-mini",
            "messages": [
                {
                    "role": "system",
                    "content": f"You are a helpful AI 3D design assistant. The user is currently viewing a 3D model: {request.model_context or 'Unknown'}. Answer their questions concisely.",
                },
                {"role": "user", "content": request.message},
            ],
        }
        headers = {
            "Authorization": f"Bearer {FREE_AI_API_KEY}",
            "Content-Type": "application/json",
        }
        try:
            resp = requests.post(FREE_AI_API_URL, json=payload, headers=headers, timeout=30)
        except Exception as e:
            print(f"chat request to FREE_AI provider failed: {e}")
            raise HTTPException(status_code=502, detail="Failed to contact configured AI provider")

        if resp.status_code >= 400:
            print(f"AI provider error ({resp.status_code}): {resp.text}")
            raise HTTPException(status_code=502, detail=f"AI provider error: {resp.status_code}")

        try:
            data = resp.json()
        except Exception:
            return {"status": "success", "message": resp.text}

        # Parse OpenAI/OpenRouter-style responses
        message_text = None
        try:
            if isinstance(data, dict) and "choices" in data and len(data["choices"]) > 0:
                choice = data["choices"][0]
                if isinstance(choice.get("message"), dict) and choice["message"].get("content"):
                    message_text = choice["message"]["content"]
                elif choice.get("text"):
                    message_text = choice.get("text")
            if not message_text:
                message_text = data.get("completion") or data.get("output") or data.get("text")
        except Exception as e:
            print(f"Error parsing AI response: {e} - raw: {data}")

        return {
            "status": "success",
            "message": message_text or "(no text in provider response)",
        }

    # If no provider configured, return a clear 503
    raise HTTPException(
        status_code=503,
        detail=("No AI provider configured. Set `GROQ_API_KEY` or `FREE_AI_API_URL`+`FREE_AI_API_KEY` in `Backend/.env`."),
    )


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
