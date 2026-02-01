"""
Test script for loading and running SmolVLA inference.
SmolVLA is a 450M parameter VLA model that runs on consumer hardware without quantization.

References:
- Model: https://huggingface.co/lerobot/smolvla_base
- Blog: https://huggingface.co/blog/smolvla
- GitHub: https://github.com/huggingface/lerobot
"""

import os
import gc
import sys
import time
import logging

# Configure verbose logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("smolvla_test.log", mode="w", encoding="utf-8"),
    ],
)

logger = logging.getLogger(__name__)

logger.info("=" * 70)
logger.info("SmolVLA Test Script - Starting")
logger.info("=" * 70)

logger.info("Importing torch...")
import_start = time.time()
import torch
logger.info(f"torch imported in {time.time() - import_start:.2f}s")

logger.info("Importing numpy and PIL...")
import numpy as np
from PIL import Image
import requests
from io import BytesIO
logger.info("Core imports complete")


def print_gpu_memory(context=""):
    """Print current GPU memory usage with context."""
    prefix = f"[{context}] " if context else ""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        free = total - allocated
        logger.info(f"{prefix}GPU Memory - Allocated: {allocated:.3f} GB | Reserved: {reserved:.3f} GB | Free: {free:.3f} GB")
    else:
        logger.info(f"{prefix}CUDA not available, using CPU")


def print_system_info():
    """Print detailed system information."""
    logger.info("-" * 70)
    logger.info("SYSTEM INFORMATION")
    logger.info("-" * 70)

    logger.info(f"Python version: {sys.version}")
    logger.info(f"Python executable: {sys.executable}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"cuDNN version: {torch.backends.cudnn.version()}")
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"  Device {i}: {props.name}")
            logger.info(f"    - Compute capability: {props.major}.{props.minor}")
            logger.info(f"    - Total memory: {props.total_memory / 1024**3:.2f} GB")
            logger.info(f"    - Multi-processor count: {props.multi_processor_count}")

    # Check lerobot
    try:
        import lerobot
        logger.info(f"lerobot version: {lerobot.__version__ if hasattr(lerobot, '__version__') else 'unknown'}")
    except ImportError as e:
        logger.error(f"lerobot not installed: {e}")

    print_gpu_memory("Initial state")
    logger.info("-" * 70)


def load_model_and_processors(device):
    """Load SmolVLA model and preprocessors."""
    logger.info("-" * 70)
    logger.info("LOADING SMOLVLA MODEL AND PROCESSORS")
    logger.info("-" * 70)

    print_gpu_memory("Before model load")

    # Import SmolVLAPolicy
    logger.info("Importing SmolVLAPolicy from lerobot...")
    import_start = time.time()
    try:
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        logger.info("Using lerobot.policies import path")
    except ImportError:
        from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        logger.info("Using lerobot.common.policies import path")
    logger.info(f"SmolVLAPolicy imported in {time.time() - import_start:.2f}s")

    # Import preprocessor factory
    logger.info("Importing preprocessor factory...")
    try:
        from lerobot.policies.factory import make_pre_post_processors
        logger.info("Using lerobot.policies.factory")
    except ImportError:
        from lerobot.common.policies.factory import make_pre_post_processors
        logger.info("Using lerobot.common.policies.factory")

    model_id = "lerobot/smolvla_base"
    logger.info(f"Loading model: {model_id}")
    logger.info("This may take a few minutes on first run (downloading model)...")

    # Load model
    load_start = time.time()
    policy = SmolVLAPolicy.from_pretrained(model_id)
    load_time = time.time() - load_start
    logger.info(f"Model loaded in {load_time:.2f}s")

    logger.info(f"Policy type: {type(policy).__name__}")
    logger.info(f"Policy config: {policy.config}")

    # Count parameters
    total_params = sum(p.numel() for p in policy.parameters())
    logger.info(f"Total parameters: {total_params:,} ({total_params / 1e6:.1f}M)")

    # Create pre/post processors
    logger.info("Creating pre/post processors...")
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        model_id,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    logger.info("Processors created")

    # Move to device
    logger.info(f"Moving model to {device}...")
    move_start = time.time()
    policy = policy.to(device)
    logger.info(f"Model moved to {device} in {time.time() - move_start:.2f}s")

    # Set to eval mode
    policy.eval()
    logger.info("Model set to eval mode")

    print_gpu_memory("After model load")

    return policy, preprocess, postprocess


def create_raw_observation(policy):
    """Create a raw observation dict (before preprocessing)."""
    logger.info("-" * 70)
    logger.info("CREATING RAW OBSERVATION")
    logger.info("-" * 70)

    # Get expected image features from policy config
    logger.info("Checking policy config for expected input features...")

    # Find image feature keys from policy config
    image_keys = []
    img_shape = (256, 256)  # H, W

    if hasattr(policy.config, 'image_features'):
        logger.info(f"Image features from config: {policy.config.image_features}")
        for key, feature in policy.config.image_features.items():
            # Convert observation.images.camera1 -> camera1
            cam_name = key.split(".")[-1]
            image_keys.append(cam_name)
            if hasattr(feature, 'shape') and len(feature.shape) >= 2:
                img_shape = (feature.shape[-2], feature.shape[-1])
            logger.info(f"  {key} -> {cam_name}")

    if not image_keys:
        image_keys = ["camera1", "camera2", "camera3"]
        logger.info(f"Using default image keys: {image_keys}")

    # Create dummy RGB images (H, W, C format for raw observation)
    img_h, img_w = img_shape
    logger.info(f"Creating dummy images: {img_h}x{img_w}x3")

    # Create a simple test pattern image
    img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    img[:img_h//3, :, 0] = 200  # Red band
    img[img_h//3:2*img_h//3, :, 1] = 200  # Green band
    img[2*img_h//3:, :, 2] = 200  # Blue band
    logger.info(f"Test image created with shape: {img.shape}, dtype: {img.dtype}")

    # Get state dimension from the policy config
    state_dim = 6  # Default
    if hasattr(policy.config, 'input_shapes'):
        try:
            state_dim = policy.config.input_shapes["observation.state"][0]
            logger.info(f"State dimension from policy config: {state_dim}")
        except Exception as e:
            logger.warning(f"Could not get state dim from config: {e}")
            logger.info(f"Using default state dimension: {state_dim}")

    # Create dummy state (joint positions)
    state = np.zeros(state_dim, dtype=np.float32)
    logger.info(f"State shape: {state.shape}, dtype: {state.dtype}")

    # Build raw observation dict
    # This format matches what comes from robot.get_observation()
    observation = {}

    # Add images (raw numpy arrays in H, W, C format)
    for cam_name in image_keys:
        observation[cam_name] = img.copy()
        logger.info(f"Added image: {cam_name} shape={img.shape}")

    # Add state
    observation["agent_pos"] = state
    logger.info(f"Added agent_pos: shape={state.shape}")

    logger.info("Raw observation dict keys:")
    for key, value in observation.items():
        if isinstance(value, np.ndarray):
            logger.info(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            logger.info(f"  {key}: {value}")

    return observation


def run_inference(policy, preprocess, postprocess, raw_obs, task, device, run_number):
    """Run inference to predict robot action."""
    logger.info("-" * 70)
    logger.info(f"INFERENCE RUN #{run_number}")
    logger.info("-" * 70)
    logger.info(f"Task: {task}")

    print_gpu_memory(f"Before inference #{run_number}")

    # Import build_inference_frame helper
    try:
        from lerobot.policies.utils import build_inference_frame
        logger.info("Using lerobot.policies.utils")
    except ImportError:
        from lerobot.common.policies.utils import build_inference_frame
        logger.info("Using lerobot.common.policies.utils")

    # Build dataset features from policy config
    logger.info("Building inference frame...")

    # Create a simple features dict based on policy config
    ds_features = {}

    # Add image features
    if hasattr(policy.config, 'image_features'):
        for key, feature in policy.config.image_features.items():
            ds_features[key] = {"dtype": "image", "shape": feature.shape}

    # Add state features
    if hasattr(policy.config, 'input_shapes'):
        if "observation.state" in policy.config.input_shapes:
            ds_features["observation.state"] = {
                "dtype": "float32",
                "shape": policy.config.input_shapes["observation.state"]
            }

    # Add action features
    if hasattr(policy.config, 'output_shapes'):
        if "action" in policy.config.output_shapes:
            ds_features["action"] = {
                "dtype": "float32",
                "shape": policy.config.output_shapes["action"]
            }

    logger.info(f"Dataset features: {ds_features}")

    # Build the frame
    frame_start = time.time()
    try:
        obs_frame = build_inference_frame(
            observation=raw_obs,
            ds_features=ds_features,
            device=device,
            task=task,
            robot_type="",
        )
        logger.info(f"Inference frame built in {time.time() - frame_start:.4f}s")
    except Exception as e:
        logger.warning(f"build_inference_frame failed: {e}")
        logger.info("Falling back to manual frame construction...")

        # Manual frame construction
        obs_frame = {}
        for key, value in raw_obs.items():
            if isinstance(value, np.ndarray):
                if len(value.shape) == 3:  # Image
                    # Convert HWC to CHW and add batch dim
                    tensor = torch.from_numpy(value).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                    obs_frame[f"observation.images.{key}"] = tensor.to(device)
                else:
                    tensor = torch.from_numpy(value).unsqueeze(0).float()
                    obs_frame["observation.state"] = tensor.to(device)
        obs_frame["task"] = [task]
        logger.info(f"Manual frame built with keys: {obs_frame.keys()}")

    # Preprocess
    logger.info("Preprocessing observation...")
    preprocess_start = time.time()
    try:
        processed_obs = preprocess(obs_frame)
        logger.info(f"Preprocessing completed in {time.time() - preprocess_start:.4f}s")
        logger.info(f"Processed observation keys: {processed_obs.keys() if hasattr(processed_obs, 'keys') else type(processed_obs)}")
    except Exception as e:
        logger.warning(f"Preprocess failed: {e}, using raw frame")
        processed_obs = obs_frame

    # Run inference
    logger.info("Running policy.select_action()...")
    inference_start = time.time()

    with torch.inference_mode():
        action = policy.select_action(processed_obs)

    inference_time = time.time() - inference_start
    logger.info(f"Inference completed in {inference_time:.4f}s")

    # Postprocess
    logger.info("Postprocessing action...")
    try:
        action = postprocess(action)
        logger.info("Postprocessing completed")
    except Exception as e:
        logger.warning(f"Postprocess failed: {e}, using raw action")

    print_gpu_memory(f"After inference #{run_number}")

    # Log action details
    logger.info(f"Action type: {type(action)}")
    if isinstance(action, torch.Tensor):
        logger.info(f"Action shape: {action.shape}")
        logger.info(f"Action dtype: {action.dtype}")
        action_np = action.cpu().numpy() if action.device.type != 'cpu' else action.numpy()
        logger.info(f"Action values: {action_np}")
    elif isinstance(action, dict):
        logger.info("Action is a dict:")
        for key, value in action.items():
            if isinstance(value, torch.Tensor):
                val_np = value.cpu().numpy() if value.device.type != 'cpu' else value.numpy()
                logger.info(f"  {key}: shape={value.shape}, values={val_np}")
            else:
                logger.info(f"  {key}: {value}")
    elif isinstance(action, np.ndarray):
        logger.info(f"Action shape: {action.shape}")
        logger.info(f"Action values: {action}")
    else:
        logger.info(f"Action: {action}")

    return action, inference_time


def main():
    """Main entry point."""
    total_start = time.time()

    logger.info("=" * 70)
    logger.info("SmolVLA Inference Test")
    logger.info("Model: 450M parameters - runs without quantization")
    logger.info("=" * 70)

    # Clear memory
    logger.info("Clearing CUDA cache...")
    torch.cuda.empty_cache()
    gc.collect()

    # Determine device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA device")
    else:
        device = torch.device("cpu")
        logger.info("CUDA not available, using CPU")

    # Print system info
    print_system_info()

    # Load model and processors
    policy, preprocess, postprocess = load_model_and_processors(device)

    # Create raw observation
    raw_obs = create_raw_observation(policy)

    # Run multiple inference tests
    logger.info("=" * 70)
    logger.info("RUNNING INFERENCE TESTS")
    logger.info("=" * 70)

    tasks = [
        "pick up the red object",
        "move the arm to the left",
        "place the object on the table",
    ]
    num_runs = len(tasks)
    logger.info(f"Number of test runs: {num_runs}")

    inference_times = []
    for i, task in enumerate(tasks, 1):
        action, inf_time = run_inference(
            policy, preprocess, postprocess, raw_obs, task, device, i
        )
        inference_times.append(inf_time)

    # Summary
    total_time = time.time() - total_start

    logger.info("=" * 70)
    logger.info("TEST SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total test time: {total_time:.2f}s")
    logger.info(f"Number of inferences: {len(inference_times)}")
    logger.info(f"Average inference time: {sum(inference_times) / len(inference_times):.4f}s")
    logger.info(f"Min inference time: {min(inference_times):.4f}s")
    logger.info(f"Max inference time: {max(inference_times):.4f}s")
    logger.info(f"First inference (includes warmup): {inference_times[0]:.4f}s")
    if len(inference_times) > 1:
        logger.info(f"Average excluding first: {sum(inference_times[1:]) / len(inference_times[1:]):.4f}s")

    print_gpu_memory("Final state")

    logger.info("=" * 70)
    logger.info("TEST COMPLETED SUCCESSFULLY")
    logger.info("=" * 70)
    logger.info(f"Log file saved to: smolvla_test.log")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Test interrupted by user (Ctrl+C)")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Test failed with error: {type(e).__name__}: {e}")
        sys.exit(1)
