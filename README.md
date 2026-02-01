# SmolVLA Test

Test script for running [SmolVLA](https://huggingface.co/lerobot/smolvla_base) inference locally.

SmolVLA is a 450M parameter Vision-Language-Action model for robotics that runs on consumer hardware without quantization.

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support (or CPU, but slower)
- ~3GB VRAM (or system RAM for CPU)

## Install uv

[uv](https://github.com/astral-sh/uv) is a fast Python package manager.

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Windows (winget):**
```bash
winget install --id=astral-sh.uv -e
```

**Linux/macOS:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Setup

```bash
# Create virtual environment
uv venv

# Activate (Windows PowerShell)
.venv\Scripts\activate

# Activate (Windows CMD)
.venv\Scripts\activate.bat

# Activate (Linux/macOS)
source .venv/bin/activate

# Install PyTorch with CUDA
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install lerobot with SmolVLA support
uv pip install "lerobot[smolvla]"

# Install remaining dependencies
uv pip install -r requirements.txt
```

## Run

```bash
python test_smolvla_simple.py
```

## What the script does

`test_smolvla_simple.py` is a minimal test to verify SmolVLA runs on your hardware:

1. **Detects device** - Uses CUDA GPU if available, otherwise CPU
2. **Loads SmolVLA** - Downloads the 450M parameter model from HuggingFace (`lerobot/smolvla_base`)
3. **Creates preprocessor** - Sets up LeRobot's preprocessing pipeline (handles image normalization and text tokenization)
4. **Builds dummy input batch**:
   - Random images for each camera (camera1, camera2, camera3)
   - Zero robot state vector (joint positions)
   - Text instruction: "pick up the red block"
5. **Runs inference** - Feeds the batch through the model
6. **Outputs action** - Prints the predicted robot action (joint commands)

The output action will be meaningless since the input is random - this just validates the model loads and runs correctly.

## Model info

| Property | Value |
|----------|-------|
| Model | `lerobot/smolvla_base` |
| Parameters | 450M |
| Architecture | SmolVLM2 (500M) backbone + Action Expert (~100M) |
| Input | 3 RGB camera images + robot state + language instruction |
| Output | Action chunks (sequence of robot joint commands) |

## Files

| File | Description |
|------|-------------|
| `test_smolvla_simple.py` | Minimal inference test script |
| `test_smolvla.py` | Verbose test script with detailed logging |
| `requirements.txt` | Python dependencies |

## References

- [SmolVLA Model](https://huggingface.co/lerobot/smolvla_base)
- [SmolVLA Blog](https://huggingface.co/blog/smolvla)
- [LeRobot GitHub](https://github.com/huggingface/lerobot)
- [SmolVLA Paper](https://arxiv.org/abs/2506.01844)
