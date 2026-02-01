"""
Test script for loading OpenVLA with NF4 4-bit quantization.
Designed to run on RTX 3060 Ti (8GB VRAM) or similar consumer GPUs.
"""

import os
import gc
import torch
import requests
from io import BytesIO
from PIL import Image

# Memory optimization before loading
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()
gc.collect()


def print_gpu_memory():
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
    else:
        print("CUDA not available")


def download_test_image():
    """Download a sample image for testing."""
    # Using a simple test image (robot arm scene from OpenVLA examples)
    url = "https://huggingface.co/openvla/openvla-7b/resolve/main/assets/example_image.png"

    print("Downloading test image...")
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
        print(f"Image downloaded: {image.size}")
        return image
    except Exception as e:
        print(f"Failed to download test image: {e}")
        print("Creating a dummy test image instead...")
        # Create a simple dummy image if download fails
        return Image.new("RGB", (256, 256), color=(128, 128, 128))


def load_model_nf4():
    """Load OpenVLA with NF4 4-bit quantization."""
    from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

    print("\n" + "=" * 60)
    print("Loading OpenVLA with NF4 4-bit quantization")
    print("=" * 60)

    # NF4 quantization configuration
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    print("\nLoading processor...")
    processor = AutoProcessor.from_pretrained(
        "openvla/openvla-7b",
        trust_remote_code=True,
    )
    print("Processor loaded.")

    print("\nLoading model (this may take a few minutes on first run)...")
    print_gpu_memory()

    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b",
        quantization_config=quant_config,
        device_map="auto",
        max_memory={0: "7.5GB", "cpu": "16GB"},
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    print("Model loaded!")
    print_gpu_memory()

    return model, processor


def run_inference(model, processor, image, instruction):
    """Run inference to predict robot action."""
    print(f"\nInstruction: '{instruction}'")

    # Format prompt (OpenVLA specific format)
    prompt = f"In: What action should the robot take to {instruction}?\nOut:"

    # Process inputs
    inputs = processor(prompt, image).to("cuda:0", dtype=torch.float16)

    # Run inference
    with torch.inference_mode():
        action = model.predict_action(
            **inputs,
            unnorm_key="bridge_orig",  # Use bridge robot normalization
            do_sample=False,
        )

    return action


def main():
    print("=" * 60)
    print("OpenVLA NF4 4-bit Quantization Test")
    print("=" * 60)

    # Check CUDA availability
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"Total VRAM: {total_mem:.2f} GB")

    print_gpu_memory()

    # Download test image
    image = download_test_image()

    # Load model
    model, processor = load_model_nf4()

    # Test instructions
    test_instructions = [
        "pick up the object",
        "move the arm to the left",
        "place the object down",
    ]

    print("\n" + "=" * 60)
    print("Running inference tests")
    print("=" * 60)

    for instruction in test_instructions:
        action = run_inference(model, processor, image, instruction)
        print(f"Predicted action (7-DoF): {action}")
        print(f"  - Position (x, y, z): {action[:3]}")
        print(f"  - Rotation (roll, pitch, yaw): {action[3:6]}")
        print(f"  - Gripper: {action[6]}")
        print()

    print("=" * 60)
    print("Test completed successfully!")
    print("=" * 60)
    print_gpu_memory()


if __name__ == "__main__":
    main()
