from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
import httpx
import os

app = FastAPI(title="AI API Hub")

HF_TOKEN = os.getenv("HF_TOKEN")
HF_URL = "https://router.huggingface.co"

MODELS = {
    "text": {
        "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
        "llama-3": "meta-llama/Meta-Llama-3-8B-Instruct",
    },
    "tts": {
        "mms-tts-fra": "facebook/mms-tts-fra",
        "bark": "suno/bark-small",
    },
    "image": {
        "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
        "flux-schnell": "black-forest-labs/FLUX.1-schnell",
    },
}

HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}


# ─── TEXT ──────────────────────────────────────────
@app.post("/inference/text/{model_id}")
async def text_inference(model_id: str, request: Request):
    hf_model = _get_model("text", model_id)
    body = await request.json()
    body["model"] = hf_model
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(f"{HF_URL}/v1/chat/completions", headers=HEADERS, json=body)

    if resp.status_code != 200:
        raise HTTPException(resp.status_code, resp.json())

    return resp.json()


# ─── TTS ───────────────────────────────────────────
@app.post("/inference/tts/{model_id}")
async def tts_inference(model_id: str, request: Request):
    hf_model = _get_model("tts", model_id)
    body = await request.json()
    body["model"] = hf_model

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(f"{HF_URL}/v1/audio/speech", headers=HEADERS, json=body)

    if resp.status_code != 200:
        raise HTTPException(resp.status_code, resp.text)

    content_type = resp.headers.get("content-type", "audio/wav")
    return Response(content=resp.content, media_type=content_type)


# ─── IMAGE ─────────────────────────────────────────
@app.post("/inference/image/{model_id}")
async def image_inference(model_id: str, request: Request):
    hf_model = _get_model("image", model_id)
    body = await request.json()
    body["model"] = hf_model

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(f"{HF_URL}/v1/images/generations", headers=HEADERS, json=body)

    if resp.status_code != 200:
        raise HTTPException(resp.status_code, resp.text)

    content_type = resp.headers.get("content-type", "image/png")
    return Response(content=resp.content, media_type=content_type)


# ─── MODELS LIST ───────────────────────────────────
@app.get("/models")
async def list_models():
    return MODELS


@app.get("/models/{task}")
async def list_models_by_task(task: str):
    if task not in MODELS:
        raise HTTPException(404, f"Task not found. Available: {list(MODELS.keys())}")
    return MODELS[task]


# ─── HELPER ────────────────────────────────────────
def _get_model(task: str, model_id: str) -> str:
    if task not in MODELS:
        raise HTTPException(404, f"Task not found. Available: {list(MODELS.keys())}")
    if model_id not in MODELS[task]:
        raise HTTPException(404, f"Model not found. Available: {list(MODELS[task].keys())}")
    return MODELS[task][model_id]
