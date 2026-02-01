"""
Raspberry Pi Client for SmolVLA Robot Control

Runs on Pi 5. Captures camera, sends to laptop for inference, executes actions.

Usage:
    python pi_client.py --server http://192.168.1.100:8000

Requirements (install on Pi):
    pip install opencv-python requests numpy lerobot
"""

import argparse
import time
import base64
import io
import logging
from dataclasses import dataclass

import cv2
import numpy as np
import requests

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    # Server (your laptop)
    server_url: str = "http://192.168.1.100:8000"

    # Camera
    camera_index: int = 0
    camera_width: int = 640
    camera_height: int = 480

    # Robot
    robot_port: str = "/dev/ttyUSB0"
    robot_id: str = "follower"

    # Task
    task: str = "pick up the red block"

    # Control loop
    max_hz: float = 10.0  # Max control frequency


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

        logger.info(f"Camera opened: index={index}, {width}x{height}")

    def capture(self) -> np.ndarray:
        """Capture a frame. Returns RGB numpy array."""
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to capture frame")
        # Convert BGR to RGB
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def capture_jpeg_base64(self, quality: int = 80) -> str:
        """Capture and encode as base64 JPEG for sending over network."""
        frame = self.capture()
        # Convert RGB back to BGR for OpenCV encoding
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # Encode as JPEG
        _, buffer = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        # Base64 encode
        return base64.b64encode(buffer).decode("utf-8")

    def close(self):
        self.cap.release()


# ============================================================================
# ROBOT (using lerobot)
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
        except ImportError:
            logger.warning("lerobot not installed, using mock robot")
            self.robot = None
        except Exception as e:
            logger.error(f"Failed to connect robot: {e}")
            self.robot = None

    def get_state(self) -> list[float]:
        """Get current joint positions."""
        if self.robot is None:
            return [0.0] * 6  # Mock state

        try:
            obs = self.robot.get_observation()
            # Extract state from observation
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
            logger.debug(f"Mock action: {action[:3]}...")
            return

        try:
            self.robot.send_action(np.array(action))
        except Exception as e:
            logger.error(f"Failed to send action: {e}")

    def disconnect(self):
        if self.robot:
            self.robot.disconnect()
            logger.info("Robot disconnected")


# ============================================================================
# INFERENCE CLIENT
# ============================================================================

class InferenceClient:
    """HTTP client for laptop inference server."""

    def __init__(self, server_url: str):
        self.server_url = server_url.rstrip("/")
        self.predict_url = f"{self.server_url}/predict"
        self.session = requests.Session()

    def health_check(self) -> bool:
        """Check if server is reachable."""
        try:
            resp = self.session.get(f"{self.server_url}/health", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def predict(self, image_base64: str, state: list[float], task: str) -> dict:
        """
        Send observation to server, get action back.

        Returns: {"action": [...], "inference_time": float, "success": bool}
        """
        payload = {
            "image": image_base64,
            "state": state,
            "task": task,
        }

        try:
            resp = self.session.post(self.predict_url, json=payload, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            return {"action": state, "inference_time": 0, "success": False}


# ============================================================================
# CONTROL LOOP
# ============================================================================

def run_control_loop(config: Config):
    """Main control loop: capture → send → receive → execute."""

    logger.info("=" * 60)
    logger.info("SmolVLA Robot Control (Pi Client)")
    logger.info("=" * 60)
    logger.info(f"Server: {config.server_url}")
    logger.info(f"Task: {config.task}")
    logger.info("=" * 60)

    # Initialize components
    camera = Camera(config.camera_index, config.camera_width, config.camera_height)
    robot = Robot(config.robot_port, config.robot_id)
    client = InferenceClient(config.server_url)

    # Connect robot
    robot.connect()

    # Check server
    logger.info("Checking server connection...")
    if not client.health_check():
        logger.error(f"Cannot reach server at {config.server_url}")
        logger.error("Make sure laptop_server.py is running!")
        return

    logger.info("Server connected!")
    logger.info("Starting control loop... (Ctrl+C to stop)")
    logger.info("-" * 60)

    loop_period = 1.0 / config.max_hz
    loop_count = 0
    total_time = 0

    try:
        while True:
            loop_start = time.time()

            # 1. Capture image
            image_b64 = camera.capture_jpeg_base64(quality=80)

            # 2. Get current state
            state = robot.get_state()

            # 3. Send to server for inference
            result = client.predict(image_b64, state, config.task)

            # 4. Execute action
            if result["success"]:
                robot.send_action(result["action"])

            # Stats
            loop_time = time.time() - loop_start
            total_time += loop_time
            loop_count += 1

            if loop_count % 10 == 0:
                avg_time = total_time / loop_count
                hz = 1.0 / avg_time if avg_time > 0 else 0
                logger.info(
                    f"Loop {loop_count}: {loop_time*1000:.0f}ms "
                    f"(avg: {avg_time*1000:.0f}ms, {hz:.1f} Hz) "
                    f"| Inference: {result.get('inference_time', 0)*1000:.0f}ms"
                )

            # Rate limiting
            elapsed = time.time() - loop_start
            if elapsed < loop_period:
                time.sleep(loop_period - elapsed)

    except KeyboardInterrupt:
        logger.info("\nStopping...")

    finally:
        camera.close()
        robot.disconnect()
        logger.info("Cleanup complete")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="SmolVLA Pi Client")
    parser.add_argument(
        "--server",
        type=str,
        default="http://192.168.1.100:8000",
        help="Laptop server URL (e.g., http://192.168.1.100:8000)",
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--port", type=str, default="/dev/ttyUSB0", help="Robot USB port")
    parser.add_argument("--robot-id", type=str, default="follower", help="Robot ID")
    parser.add_argument("--task", type=str, default="pick up the red block", help="Task instruction")
    parser.add_argument("--hz", type=float, default=10.0, help="Max control frequency")

    args = parser.parse_args()

    config = Config(
        server_url=args.server,
        camera_index=args.camera,
        robot_port=args.port,
        robot_id=args.robot_id,
        task=args.task,
        max_hz=args.hz,
    )

    run_control_loop(config)


if __name__ == "__main__":
    main()
