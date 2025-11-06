import io, os, base64
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.staticfiles import StaticFiles
from ai_pipeline import DeclutterPipeline

app = FastAPI(title="Declutter API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

# Init pipeline
pipeline = DeclutterPipeline()

# ---------------- API ROUTES ----------------
@app.post("/api/declutter")
async def declutter(file: UploadFile = File(...), prompt: str = Form(default="")):
    img_bytes = await file.read()
    result = pipeline.run(img_bytes, user_prompt=prompt)

    buf = io.BytesIO()
    result["final_image"].save(buf, format="JPEG", quality=92)
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return JSONResponse({
        "prompt_used": result["prompt"],
        "detected_classes": result["classes"],
        "image_base64": f"data:image/jpeg;base64,{img_b64}"
    })

@app.get("/health")
def health():
    return {"ok": True}

# ---------------- STATIC FRONTEND ----------------
# Serve / -> index.html inside ../web
web_dir = os.path.join(os.path.dirname(__file__), "..", "web")
app.mount("/", StaticFiles(directory=web_dir, html=True), name="static")

