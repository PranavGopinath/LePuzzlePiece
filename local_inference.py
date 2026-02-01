"""
Local SmolVLA Inference for SO101 Robot

Runs everything on your laptop - camera capture, model inference, and robot control.
No Raspberry Pi required.

Usage:
    python local_inference.py --port COM3 --camera 0 --task "pick up the red block"

Requirements:
    pip install torch torchvision lerobot opencv-python
"""

import argparse
import time
import logging
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from PIL import Image

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    # Camera
    camera_index: int = 0
    camera_width: int = 640
    camera_height: int = 480

    # Robot
    robot_port: str = "COM3"
    robot_id: str = "follower"
    no_robot: bool = False  # Run without robot for testing

    # Model
    model_id: str = "lerobot/smolvla_base"
    device: str = "cuda"

    # Task
    task: str = "pick up the red block"

    # Control loop
    hz: float = 10.0


# ============================================================================
# CAMERA
# ============================================================================

class Camera:
    """Simple OpenCV camera wrapper."""

    def __init__(self, index: int, width: int, height: int):
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {index}")

        # Read actual resolution (may differ from requested)
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"Camera opened: index={index}, {actual_w}x{actual_h}")

    def capture(self) -> np.ndarray:
        """Capture a frame. Returns RGB numpy array (H, W, 3)."""
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to capture frame")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def close(self):
        self.cap.release()
        logger.info("Camera closed")


# ============================================================================
# ROBOT
# ============================================================================

class Robot:
    """SO101 robot arm wrapper using lerobot."""

    def __init__(self, port: str, robot_id: str):
        self.port = port
        self.robot_id = robot_id
        self.robot = None

    def connect(self):
        """Connect to robot arm."""
        try:
            from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

            config = SO101FollowerConfig(port=self.port, id=self.robot_id)
            self.robot = SO101Follower(config)
            self.robot.connect()
            logger.info(f"Robot connected: {self.port}")
        except ImportError as e:
            logger.error(f"lerobot not installed: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to connect robot: {e}")
            raise

    def get_state(self) -> list[float]:
        """Get current joint positions (6 values)."""
        if self.robot is None:
            return [0.0] * 6

        try:
            obs = self.robot.get_observation()
            if "observation.state" in obs:
                state = obs["observation.state"]
                if hasattr(state, "tolist"):
                    return state.tolist()
                return list(state)
            return [0.0] * 6
        except Exception as e:
            logger.error(f"Failed to get state: {e}")
            return [0.0] * 6

    def send_action(self, action: list[float]):
        """Send action (joint positions) to robot."""
        if self.robot is None:
            return

        try:
            self.robot.send_action(np.array(action))
        except Exception as e:
            logger.error(f"Failed to send action: {e}")

    def disconnect(self):
        if self.robot:
            self.robot.disconnect()
            logger.info("Robot disconnected")


class MockRobot:
    """Mock robot for testing without hardware."""

    def __init__(self):
        self.state = [0.0] * 6

    def connect(self):
        logger.info("Mock robot connected (no hardware)")

    def get_state(self) -> list[float]:
        return self.state.copy()

    def send_action(self, action: list[float]):
        self.state = action[:6] if len(action) >= 6 else action + [0.0] * (6 - len(action))

    def disconnect(self):
        logger.info("Mock robot disconnected")


# ============================================================================
# MODEL
# ============================================================================

class SmolVLAModel:
    """SmolVLA model wrapper for inference."""

    def __init__(self, model_id: str, device: str):
        self.model_id = model_id
        self.device = device
        self.policy = None
        self.preprocess = None
        self.postprocess = None
        self.img_keys = None
        self.img_shape = None

    def load(self):
        """Load model and create preprocessors."""
        logger.info("=" * 60)
        logger.info("Loading SmolVLA Model")
        logger.info("=" * 60)

        # Check device
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            self.device = "cpu"

        logger.info(f"Device: {self.device}")

        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.policies.factory import make_pre_post_processors

        # Load model
        logger.info(f"Loading {self.model_id}...")
        self.policy = SmolVLAPolicy.from_pretrained(self.model_id)
        self.policy.to(self.device)
        self.policy.eval()

        params = sum(p.numel() for p in self.policy.parameters())
        logger.info(f"Model loaded: {params / 1e6:.0f}M parameters")

        # Create preprocessors
        self.preprocess, self.postprocess = make_pre_post_processors(
            self.policy.config,
            self.model_id,
            preprocessor_overrides={"device_processor": {"device": self.device}},
        )

        # Get expected shapes
        self.img_keys = list(self.policy.config.image_features.keys())
        self.img_shape = self.policy.config.image_features[self.img_keys[0]].shape

        logger.info(f"Image keys: {self.img_keys}")
        logger.info(f"Image shape (C, H, W): {self.img_shape}")

        if self.device == "cuda":
            logger.info(f"GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

        logger.info("=" * 60)
        logger.info("Model ready!")
        logger.info("=" * 60)

    def predict(self, image_rgb: np.ndarray, state: list[float], task: str) -> list[float]:
        """
        Run inference.

        Args:
            image_rgb: RGB image as numpy array (H, W, 3)
            state: Robot joint positions (6 floats)
            task: Language instruction string

        Returns:
            action: Predicted joint positions (list of floats)
        """
        # Resize image to expected shape
        target_h, target_w = self.img_shape[1], self.img_shape[2]
        pil_img = Image.fromarray(image_rgb).resize((target_w, target_h))
        image_resized = np.array(pil_img)

        # Convert to tensor: (H, W, C) -> (1, C, H, W), normalized to [0, 1]
        image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        image_tensor = image_tensor.to(self.device)

        # Build batch - same image for all camera keys
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
        if isinstance(action, list) and len(action) > 0 and isinstance(action[0], list):
            action = action[0]

        return action


# ============================================================================
# CONTROL LOOP
# ============================================================================

def run_control_loop(config: Config):
    """Main control loop: capture -> inference -> execute."""

    logger.info("=" * 60)
    logger.info("SmolVLA Local Inference")
    logger.info("=" * 60)
    logger.info(f"Camera: index {config.camera_index}")
    logger.info(f"Robot: {config.robot_port} (disabled: {config.no_robot})")
    logger.info(f"Task: {config.task}")
    logger.info(f"Target Hz: {config.hz}")
    logger.info("=" * 60)

    # Initialize camera
    camera = Camera(config.camera_index, config.camera_width, config.camera_height)

    # Initialize robot
    if config.no_robot:
        robot = MockRobot()
    else:
        robot = Robot(config.robot_port, config.robot_id)

    robot.connect()

    # Load model
    model = SmolVLAModel(config.model_id, config.device)
    model.load()

    # Control loop
    logger.info("-" * 60)
    logger.info("Starting control loop... (Ctrl+C to stop)")
    logger.info("-" * 60)

    period = 1.0 / config.hz
    loop_count = 0
    total_inference_time = 0

    try:
        while True:
            loop_start = time.time()

            # 1. Capture image
            image = camera.capture()

            # 2. Get robot state
            state = robot.get_state()

            # 3. Run inference
            inference_start = time.time()
            action = model.predict(image, state, config.task)
            inference_time = time.time() - inference_start
            total_inference_time += inference_time

            # 4. Execute action
            robot.send_action(action)

            # Stats
            loop_time = time.time() - loop_start
            loop_count += 1

            if loop_count % 10 == 0:
                avg_inference = total_inference_time / loop_count
                actual_hz = 1.0 / loop_time if loop_time > 0 else 0
                logger.info(
                    f"Loop {loop_count}: {loop_time*1000:.0f}ms | "
                    f"Inference: {inference_time*1000:.0f}ms (avg: {avg_inference*1000:.0f}ms) | "
                    f"Hz: {actual_hz:.1f}"
                )

            # 5. Rate limiting
            elapsed = time.time() - loop_start
            if elapsed < period:
                time.sleep(period - elapsed)

    except KeyboardInterrupt:
        logger.info("\nStopping...")

    finally:
        camera.close()
        robot.disconnect()

        # Final stats
        if loop_count > 0:
            avg_inference = total_inference_time / loop_count
            logger.info(f"Total loops: {loop_count}")
            logger.info(f"Average inference: {avg_inference*1000:.1f}ms")

        logger.info("Cleanup complete")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SmolVLA Local Inference for SO101 Robot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with robot on COM3
  python local_inference.py --port COM3 --camera 0 --task "pick up the red block"

  # Test without robot hardware
  python local_inference.py --no-robot --camera 0

  # Linux with USB serial
  python local_inference.py --port /dev/ttyUSB0 --camera 0

  # Use CPU instead of GPU
  python local_inference.py --device cpu --port COM3
        """,
    )

    parser.add_argument(
        "--port",
        type=str,
        default="COM3",
        help="Robot USB port (e.g., COM3 on Windows, /dev/ttyUSB0 on Linux)",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index (default: 0)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="pick up the red block",
        help="Task instruction for the model",
    )
    parser.add_argument(
        "--hz",
        type=float,
        default=10.0,
        help="Target control frequency in Hz (default: 10)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for model inference (default: cuda)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="lerobot/smolvla_base",
        help="Model ID to load (default: lerobot/smolvla_base)",
    )
    parser.add_argument(
        "--no-robot",
        action="store_true",
        help="Run without robot hardware (for testing camera and model)",
    )
    parser.add_argument(
        "--robot-id",
        type=str,
        default="follower",
        help="Robot ID (default: follower)",
    )

    args = parser.parse_args()

    config = Config(
        camera_index=args.camera,
        robot_port=args.port,
        robot_id=args.robot_id,
        no_robot=args.no_robot,
        model_id=args.model,
        device=args.device,
        task=args.task,
        hz=args.hz,
    )

    run_control_loop(config)


if __name__ == "__main__":
    main()
