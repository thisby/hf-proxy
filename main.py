from fastapi import FastAPI, Request, HTTPException
import httpx
import os

app = FastAPI()

HF_TOKEN = os.getenv("HF_TOKEN")

# Modèles supportés - tu ajoutes au fur et à mesure
MODELS = {
    "llama3-8b": "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
}

@app.post("/inference/{model_id}")
async def inference(model_id: str, request: Request):
    if model_id not in MODELS:
        raise HTTPException(404, f"Model not found. Available: {list(MODELS.keys())}")
    
    body = await request.json()
    hf_model = MODELS[model_id]
    
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            f"https://api-inference.huggingface.co/models/{hf_model}",
            headers={"Authorization": f"Bearer {HF_TOKEN}"},
            json=body
        )
    
    if resp.status_code != 200:
        raise HTTPException(resp.status_code, resp.json())
    
    return resp.json()

@app.get("/models")
async def list_models():
    return {"models": list(MODELS.keys())}

@app.get("/health")
async def health():
    return {"status": "ok"}
