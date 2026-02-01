"""
Laptop GPU Server for SmolVLA Inference

Runs on your laptop with GPU. Receives images from Pi, runs inference, returns actions.

Usage:
    python laptop_server.py

Then Pi connects to: http://<your-laptop-ip>:8000/predict
"""

import time
import base64
import io
import logging

import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_ID = "lerobot/smolvla_base"
HOST = "0.0.0.0"  # Listen on all interfaces
PORT = 8000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================================
# GLOBAL MODEL (loaded once at startup)
# ============================================================================

policy = None
preprocess = None
postprocess = None
img_keys = None
img_shape = None
state_shape = None


def load_model():
    """Load SmolVLA model and preprocessors."""
    global policy, preprocess, postprocess, img_keys, img_shape, state_shape

    logger.info("=" * 60)
    logger.info("Loading SmolVLA Model")
    logger.info("=" * 60)
    logger.info(f"Device: {DEVICE}")

    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors

    # Load model
    logger.info(f"Loading {MODEL_ID}...")
    policy = SmolVLAPolicy.from_pretrained(MODEL_ID)
    policy.to(DEVICE)
    policy.eval()

    params = sum(p.numel() for p in policy.parameters())
    logger.info(f"Model loaded: {params / 1e6:.0f}M parameters")

    # Create preprocessors
    logger.info("Creating preprocessors...")
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        MODEL_ID,
        preprocessor_overrides={"device_processor": {"device": DEVICE}},
    )

    # Get expected shapes
    img_keys = list(policy.config.image_features.keys())
    img_shape = policy.config.image_features[img_keys[0]].shape

    if hasattr(policy.config, "robot_state_feature") and policy.config.robot_state_feature:
        state_shape = policy.config.robot_state_feature.shape
    else:
        state_shape = (6,)

    logger.info(f"Image keys: {img_keys}")
    logger.info(f"Image shape: {img_shape}")
    logger.info(f"State shape: {state_shape}")

    if DEVICE == "cuda":
        logger.info(f"GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    logger.info("=" * 60)
    logger.info("Model ready!")
    logger.info("=" * 60)


# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(title="SmolVLA Inference Server")


class PredictRequest(BaseModel):
    image: str  # Base64 encoded JPEG
    state: list[float]  # Joint positions
    task: str  # Language instruction


class PredictResponse(BaseModel):
    action: list[float]
    inference_time: float
    success: bool


@app.on_event("startup")
async def startup():
    """Load model when server starts."""
    load_model()


@app.get("/")
def root():
    return {"status": "SmolVLA server running", "device": DEVICE}


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "model": MODEL_ID}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """
    Run inference on image + state + task.

    Returns action (joint positions to execute).
    """
    try:
        start_time = time.time()

        # Decode image
        image_bytes = base64.b64decode(request.image)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Resize to expected shape (W, H)
        target_w, target_h = img_shape[2], img_shape[1]
        image = image.resize((target_w, target_h))

        # Convert to tensor (C, H, W)
        image_np = np.array(image)
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        image_tensor = image_tensor.to(DEVICE)

        # Build batch (duplicate image for all camera keys)
        batch = {key: image_tensor for key in img_keys}
        batch["observation.state"] = torch.tensor(
            [request.state], dtype=torch.float32, device=DEVICE
        )
        batch["task"] = [request.task]

        # Preprocess
        processed = preprocess(batch)

        # Inference
        with torch.inference_mode():
            action = policy.select_action(processed)

        # Postprocess
        action = postprocess(action)

        # Convert to list
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy().tolist()
        elif isinstance(action, np.ndarray):
            action = action.tolist()

        # Flatten if nested
        if isinstance(action, list) and len(action) > 0 and isinstance(action[0], list):
            action = action[0]

        inference_time = time.time() - start_time

        logger.info(f"Inference: {inference_time*1000:.1f}ms | Action: {action[:3]}...")

        return PredictResponse(
            action=action,
            inference_time=inference_time,
            success=True,
        )

    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║              SmolVLA Inference Server (Local GPU)            ║
╠══════════════════════════════════════════════════════════════╣
║  This server runs on your laptop with GPU.                   ║
║  The Raspberry Pi sends images and receives actions.         ║
║                                                              ║
║  Find your IP with: ipconfig (Windows) or ip addr (Linux)    ║
║  Pi connects to: http://<your-ip>:8000/predict               ║
╚══════════════════════════════════════════════════════════════╝
""")

    uvicorn.run(app, host=HOST, port=PORT)
