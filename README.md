# Socratic AI Tutor, a U of T TMI project
## Requirements

- Node.js 22.12 or newer and npm
- Python 3.11 or newer and [uv](https://docs.astral.sh/uv/)

## Install and run

```bash
npm install
uv sync --project backend --all-groups
cp .env.example .env
npm run dev
```

Open <http://localhost:5173>. The backend runs at <http://localhost:8000>; its health check is <http://localhost:8000/api/health>.

The interface and health check work before a model is configured. To enable chat, set `LLM_BASE_URL` to the endpoint's Chat Completions API base URL, `LLM_MODEL` to its model or deployment name, and `LLM_API_KEY` if the endpoint requires authentication. These settings stay in the Python backend; restart the services after changing them. Never prefix frontend variables with `VITE_`, because Vite exposes those values to the browser.

## Checks

```bash
npm run build
npm run typecheck
npm run lint
npm test
```
