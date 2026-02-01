"""
Modal Server for SmolVLA Inference

Deploys SmolVLA model to Modal's GPU cloud for low-latency inference.
The Raspberry Pi sends images and receives robot actions via HTTP.

Deploy with:
    modal deploy modal_server.py

Test locally:
    modal serve modal_server.py
"""

import modal

# Create Modal app
app = modal.App("smolvla-inference")

# Define the container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "torchvision",
        "lerobot",
        "fastapi[standard]",
        "pillow",
        "numpy",
    )
)


@app.cls(
    image=image,
    gpu="T4",  # Options: "T4", "A10G", "A100" (T4 is cheapest)
    container_idle_timeout=300,  # Keep warm for 5 min
    allow_concurrent_inputs=10,
)
class SmolVLAInference:
    """SmolVLA model served on Modal GPU."""

    @modal.enter()
    def load_model(self):
        """Load model when container starts (cached for subsequent requests)."""
        import torch
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.policies.factory import make_pre_post_processors

        print("Loading SmolVLA model...")

        self.device = torch.device("cuda")
        self.model_id = "lerobot/smolvla_base"

        # Load policy
        self.policy = SmolVLAPolicy.from_pretrained(self.model_id)
        self.policy.to(self.device)
        self.policy.eval()

        # Create preprocessors
        self.preprocess, self.postprocess = make_pre_post_processors(
            self.policy.config,
            self.model_id,
            preprocessor_overrides={"device_processor": {"device": str(self.device)}},
        )

        # Get expected shapes
        self.img_keys = list(self.policy.config.image_features.keys())
        self.img_shape = self.policy.config.image_features[self.img_keys[0]].shape

        if hasattr(self.policy.config, 'robot_state_feature') and self.policy.config.robot_state_feature:
            self.state_shape = self.policy.config.robot_state_feature.shape
        else:
            self.state_shape = (6,)

        print(f"Model loaded! Image keys: {self.img_keys}, State shape: {self.state_shape}")

    @modal.method()
    def predict(self, image_base64: str, state: list, task: str) -> dict:
        """
        Run inference on a single observation.

        Args:
            image_base64: Base64 encoded JPEG image from camera
            state: Current robot joint positions [j1, j2, j3, j4, j5, j6]
            task: Language instruction ("pick up the red block")

        Returns:
            dict with 'action' (list of 6 floats) and 'inference_time'
        """
        import time
        import base64
        import io
        import torch
        import numpy as np
        from PIL import Image

        start_time = time.time()

        # Decode image
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize((self.img_shape[2], self.img_shape[1]))  # W, H

        # Convert to tensor (C, H, W)
        image_np = np.array(image)
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        image_tensor = image_tensor.to(self.device)

        # Build batch (duplicate image for all camera keys)
        batch = {key: image_tensor for key in self.img_keys}
        batch["observation.state"] = torch.tensor([state], dtype=torch.float32, device=self.device)
        batch["task"] = [task]

        # Preprocess
        processed = self.preprocess(batch)

        # Inference
        with torch.inference_mode():
            action = self.policy.select_action(processed)

        # Postprocess
        action = self.postprocess(action)

        # Convert to list
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy().tolist()
        elif isinstance(action, np.ndarray):
            action = action.tolist()

        # Flatten if nested
        if isinstance(action, list) and len(action) == 1:
            action = action[0]

        inference_time = time.time() - start_time

        return {
            "action": action,
            "inference_time": inference_time,
            "success": True,
        }


# FastAPI web endpoint
@app.function(image=image)
@modal.asgi_app()
def web_app():
    """FastAPI endpoint for HTTP access from Raspberry Pi."""
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel

    api = FastAPI(title="SmolVLA Inference API")
    model = SmolVLAInference()

    class PredictRequest(BaseModel):
        image: str  # Base64 encoded JPEG
        state: list[float]  # 6 joint positions
        task: str  # Language instruction

    class PredictResponse(BaseModel):
        action: list[float]
        inference_time: float
        success: bool

    @api.get("/health")
    def health():
        return {"status": "ok"}

    @api.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest):
        try:
            result = model.predict.remote(
                image_base64=request.image,
                state=request.state,
                task=request.task,
            )
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return api


# CLI for testing
@app.local_entrypoint()
def main():
    """Test the model locally."""
    import base64
    import numpy as np
    from PIL import Image
    import io

    print("Testing SmolVLA inference...")

    # Create dummy image
    dummy_image = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
    buffer = io.BytesIO()
    dummy_image.save(buffer, format="JPEG")
    image_base64 = base64.b64encode(buffer.getvalue()).decode()

    # Test inference
    model = SmolVLAInference()
    result = model.predict.remote(
        image_base64=image_base64,
        state=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        task="pick up the red block",
    )

    print(f"Result: {result}")
