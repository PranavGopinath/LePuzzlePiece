# Local Laptop Inference with S0101 Robot

Run SmolVLA inference directly on your laptop with GPU - no Raspberry Pi required.

## Overview

This guide shows how to control the SO101 robot arm using SmolVLA vision-language-action model running entirely on your laptop. The setup is simpler than the Pi+Laptop distributed architecture.

```
[USB Camera] --> [Laptop GPU] --> [SO101 Robot Arm]
                   SmolVLA
```

---

## 1. Prerequisites

### Hardware
- **Laptop with NVIDIA GPU** (RTX 3060+ recommended, 6GB+ VRAM)
- **SO101 robot arm** (6-DOF follower)
- **USB camera** (any webcam, 640x480 minimum)

### Software
```bash
# Python 3.10+
python --version

# PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# LeRobot
pip install lerobot

# OpenCV for camera
pip install opencv-python

# Verify CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## 2. Hardware Connections

### Robot Connection
Connect the SO101 robot arm via USB. The USB-to-serial adapter creates a COM port.

| OS      | Port Example     | How to Find                          |
|---------|------------------|--------------------------------------|
| Windows | `COM3`, `COM4`   | Device Manager > Ports (COM & LPT)   |
| Linux   | `/dev/ttyUSB0`   | `ls /dev/ttyUSB*`                    |
| macOS   | `/dev/tty.usb*`  | `ls /dev/tty.usb*`                   |

### Camera Connection
Plug in your USB webcam. It's typically index `0` if it's the only camera.

```python
# Test camera detection
import cv2
cap = cv2.VideoCapture(0)
print(f"Camera opened: {cap.isOpened()}")
cap.release()
```

---

## 3. Robot Calibration (One-Time Setup)

Before first use, calibrate the robot's servo positions:

```bash
# Windows
python -m lerobot.calibrate --robot.type=so101_follower --robot.port=COM3

# Linux
python -m lerobot.calibrate --robot.type=so101_follower --robot.port=/dev/ttyUSB0
```

Follow the on-screen instructions to move joints to their limits. Calibration data is saved automatically.

---

## 4. Getting Images from Camera

```python
import cv2
import numpy as np

class Camera:
    """Simple OpenCV camera wrapper."""

    def __init__(self, index: int = 0, width: int = 640, height: int = 480):
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {index}")

    def capture(self) -> np.ndarray:
        """Capture a frame. Returns RGB numpy array (H, W, 3)."""
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to capture frame")
        # OpenCV captures BGR, convert to RGB
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def close(self):
        self.cap.release()
```

Usage:
```python
camera = Camera(index=0)
image_rgb = camera.capture()  # Shape: (480, 640, 3), dtype: uint8
print(f"Image shape: {image_rgb.shape}")
camera.close()
```

---

## 5. Getting Robot State

```python
import numpy as np

class Robot:
    """SO101 robot arm wrapper using lerobot."""

    def __init__(self, port: str, robot_id: str = "follower"):
        self.port = port
        self.robot_id = robot_id
        self.robot = None

    def connect(self):
        from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

        config = SO101FollowerConfig(port=self.port, id=self.robot_id)
        self.robot = SO101Follower(config)
        self.robot.connect()

    def get_state(self) -> list[float]:
        """Get current joint positions (6 values)."""
        obs = self.robot.get_observation()
        if "observation.state" in obs:
            state = obs["observation.state"]
            return state.tolist() if hasattr(state, "tolist") else list(state)
        return [0.0] * 6

    def send_action(self, action: list[float]):
        """Send joint positions to robot."""
        self.robot.send_action(np.array(action))

    def disconnect(self):
        if self.robot:
            self.robot.disconnect()
```

### Joint Mapping

| Joint | Name         | Description                    |
|-------|--------------|--------------------------------|
| j1    | Base         | Base rotation (left/right)     |
| j2    | Shoulder     | Shoulder pitch (up/down)       |
| j3    | Elbow        | Elbow pitch                    |
| j4    | Wrist Pitch  | Wrist up/down                  |
| j5    | Wrist Roll   | Wrist rotation                 |
| j6    | Gripper      | Open/close gripper             |

Usage:
```python
robot = Robot(port="COM3")
robot.connect()

state = robot.get_state()
print(f"Joint positions: {state}")
# Example: [0.0, -45.0, 90.0, 0.0, 0.0, 50.0]

robot.disconnect()
```

---

## 6. Loading SmolVLA Model

```python
import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors

# Select device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load model
model_id = "lerobot/smolvla_base"
policy = SmolVLAPolicy.from_pretrained(model_id)
policy.to(device)
policy.eval()

print(f"Model loaded: {sum(p.numel() for p in policy.parameters()) / 1e6:.0f}M parameters")

# Create pre/post processors
preprocess, postprocess = make_pre_post_processors(
    policy.config,
    model_id,
    preprocessor_overrides={"device_processor": {"device": device}},
)

# Get expected input shapes
img_keys = list(policy.config.image_features.keys())
img_shape = policy.config.image_features[img_keys[0]].shape  # (C, H, W)

print(f"Image keys: {img_keys}")
print(f"Expected image shape: {img_shape}")
```

---

## 7. Running Inference

```python
import torch
import numpy as np

def run_inference(policy, preprocess, postprocess, image_rgb, state, task, device, img_keys, img_shape):
    """
    Run SmolVLA inference.

    Args:
        image_rgb: RGB image as numpy array (H, W, 3)
        state: Robot joint positions (6 floats)
        task: Language instruction string

    Returns:
        action: Predicted joint positions (6 floats)
    """
    # Resize image to expected shape
    from PIL import Image
    target_h, target_w = img_shape[1], img_shape[2]
    pil_img = Image.fromarray(image_rgb).resize((target_w, target_h))
    image_resized = np.array(pil_img)

    # Convert to tensor: (H, W, C) -> (1, C, H, W), normalized to [0, 1]
    image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    image_tensor = image_tensor.to(device)

    # Build batch - same image for all camera keys
    batch = {key: image_tensor for key in img_keys}
    batch["observation.state"] = torch.tensor([state], dtype=torch.float32, device=device)
    batch["task"] = [task]

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

    return action
```

Usage:
```python
# Single inference
action = run_inference(
    policy, preprocess, postprocess,
    image_rgb=camera.capture(),
    state=robot.get_state(),
    task="pick up the red block",
    device=device,
    img_keys=img_keys,
    img_shape=img_shape,
)
print(f"Predicted action: {action}")
```

---

## 8. Complete Control Loop

See `local_inference.py` for a full working example. Basic structure:

```python
import time

def control_loop(camera, robot, policy, preprocess, postprocess,
                 task, device, img_keys, img_shape, hz=10.0):
    """Main control loop at specified frequency."""

    period = 1.0 / hz
    print(f"Starting control loop at {hz} Hz...")
    print("Press Ctrl+C to stop")

    try:
        while True:
            loop_start = time.time()

            # 1. Capture image
            image = camera.capture()

            # 2. Get robot state
            state = robot.get_state()

            # 3. Run inference
            action = run_inference(
                policy, preprocess, postprocess,
                image, state, task,
                device, img_keys, img_shape
            )

            # 4. Execute action
            robot.send_action(action)

            # 5. Rate limiting
            elapsed = time.time() - loop_start
            if elapsed < period:
                time.sleep(period - elapsed)

    except KeyboardInterrupt:
        print("\nStopped by user")

    finally:
        camera.close()
        robot.disconnect()
```

---

## 9. Quick Start

```bash
# 1. Calibrate robot (one-time)
python -m lerobot.calibrate --robot.type=so101_follower --robot.port=COM3

# 2. Run local inference
python local_inference.py --port COM3 --camera 0 --task "pick up the red block" --hz 10
```

---

## 10. Troubleshooting

### USB Port Not Found
```bash
# Windows: Check Device Manager > Ports (COM & LPT)
# Linux:
ls /dev/ttyUSB*
# Add user to dialout group if permission denied:
sudo usermod -a -G dialout $USER
# Log out and back in
```

### Camera Not Opening
```python
# List available cameras
import cv2
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera {i}: available")
        cap.release()
```

### CUDA Out of Memory
```python
# Check GPU memory usage
import torch
print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

# Clear cache
torch.cuda.empty_cache()
```

SmolVLA base requires ~4GB VRAM. If you have less, try:
- Close other GPU applications
- Reduce image resolution
- Use CPU (slower): `--device cpu`

### Robot Not Responding
1. Check USB connection is secure
2. Verify correct COM port
3. Ensure robot is powered on
4. Try re-running calibration

### Slow Inference
Expected performance on RTX 3060:
- ~50-100ms per inference
- ~10-20 Hz control loop

If slower:
- Ensure CUDA is being used (check device output)
- Close other GPU applications
- Reduce camera resolution

---

## Files Reference

| File | Description |
|------|-------------|
| `local_inference.py` | Complete runnable script |
| `pi_client.py` | Pi-based client (alternative setup) |
| `laptop_server.py` | Server for Pi+Laptop setup |
| `test_smolvla_simple.py` | Minimal model test |

---

## Next Steps

1. **Test camera only**: Run `local_inference.py --no-robot` to test camera and model
2. **Test robot only**: Run calibration to verify robot communication
3. **Full loop**: Run complete pipeline with robot connected
4. **Custom tasks**: Try different task instructions to see model behavior
