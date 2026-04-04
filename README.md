# Concept-2-3D

Concept-2-3D is a full-stack 3D generation and discovery project with:

- A FastAPI backend for intent parsing, model search, AI chat, reviews, and image-to-3D generation.
- A React + Vite frontend for searching, viewing, and interacting with 3D assets.
- ML support files for TripoSR-based image-to-3D workflows.

## Monorepo layout

- Backend: FastAPI service and APIs.
- Frontend/3d-models-frontend: React app (Vite).
- ML/3d-models-ml: TripoSR helpers, shims, and model-weight utilities.
- scripts: one-command virtual environment setup for Windows and Linux/macOS.

## Features

- Intent analysis endpoint for prompt understanding.
- Multi-source model search with cache support.
- AI chat endpoint with provider fallback behavior.
- Review and rating APIs for models.
- Image-to-3D generation endpoints:
	- Synchronous: api/generate_from_image
	- Asynchronous with polling: api/generate_from_image_async and api/generate_status/{job_id}

## Prerequisites

- Python 3.10+ recommended
- Node.js 18+ and npm
- Git

## Quick start

### 1) Clone and enter project

```powershell
git clone https://github.com/GuruMachanica/Concept-2-3D.git
cd Concept-2-3D
```

### 2) Create a single Python virtual environment

Windows PowerShell:

```powershell
.\scripts\setup_venv.ps1
```

Linux/macOS:

```bash
./scripts/setup_venv.sh
```

### 3) Configure backend environment

Copy Backend/.env.example to Backend/.env and update values as needed.

At minimum, configure the providers you plan to use (for example GROQ or FREE_AI settings).

### 4) Install frontend dependencies

```powershell
cd Frontend/3d-models-frontend
npm install
cd ../..
```

### 5) Run backend

From repository root:

```powershell
.venv\Scripts\python.exe -m uvicorn Backend.main:app --host 0.0.0.0 --port 8011 --reload
```

Why port 8011:
- Frontend Vite proxy defaults to http://localhost:8011 for both /api and /models.

### 6) Run frontend

In a second terminal:

```powershell
cd Frontend/3d-models-frontend
npm run dev
```

Open the local Vite URL shown in terminal (usually http://localhost:5173).

## Backend API overview

Common endpoints:

- POST api/intent
- POST api/search
- POST api/chat
- POST api/cache/clear
- POST api/generate_from_image
- POST api/generate_from_image_async
- GET api/generate_status/{job_id}
- POST api/reviews/submit
- GET api/reviews/{model_id}
- GET api/reviews/{model_id}/summary
- GET api/reviews/{model_id}/user/{user_id}
- POST api/labels/position-from-image

When backend is running locally on port 8011, docs are available at:

- http://127.0.0.1:8011/docs

## Notes on image-to-3D

- Runtime outputs are written under Backend/models.
- Large local model-weight artifacts should stay out of git.
- TripoSR generation failures are surfaced through async status error fields and latest run logs.

## Development tips

- Lint backend (if ruff is installed in venv):

```powershell
.venv\Scripts\python.exe -m ruff check Backend
```

- Lint frontend:

```powershell
cd Frontend/3d-models-frontend
npm run lint
```

## Troubleshooting

- Frontend cannot reach backend:
	- Ensure backend is running on port 8011, or set BACKEND environment variable for Vite proxy.
- Image-to-3D fails immediately:
	- Check Backend/models latest tripo run folder and run.log.
- AI chat returns provider errors:
	- Verify GROQ or FREE_AI environment configuration in Backend/.env.

## Related docs

- README_SETUP_VENV.md for venv setup details.