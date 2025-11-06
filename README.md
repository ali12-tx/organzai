# AI Declutter App

End-to-end pipeline: **LLM (optional) → YOLOv8 (seg) → Stable Diffusion Inpainting** exposed via FastAPI, plus a simple static web UI.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt
uvicorn backend.main:app --host 0.0.0.0 --port 8000
# open web/index.html
```

## Notes
- First run downloads model weights.
- For auto LLM prompts, set `OPENAI_API_KEY` and enable the vision call stub in `ai_pipeline.py`.
- GPU strongly recommended.
