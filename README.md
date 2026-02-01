# SmolVLA Fine-tuning Pipeline

End-to-end pipeline for collecting teleoperation data and fine-tuning [SmolVLA](https://huggingface.co/lerobot/smolvla_base) on your SO101 robot.

## Overview

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Calibrate  │ -> │   Collect   │ -> │  Fine-tune  │ -> │   Deploy    │
│   Robot     │    │    Data     │    │   SmolVLA   │    │   Policy    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

## How It Works

### Leader-Follower Teleoperation

```
    YOU                          ROBOT
    ┌──────────┐                ┌──────────┐
    │  Leader  │  ──mirrors──>  │ Follower │  <── Camera
    │   Arm    │                │   Arm    │
    └──────────┘                └──────────┘
         │                            │
         └── Your hand movements ─────┘
                are recorded as
              training data
```

1. **Leader arm**: You physically move this with your hand
2. **Follower arm**: Mirrors leader's movements in real-time
3. **Camera**: Records what the follower sees during the task
4. **Data saved**: Joint positions + camera frames + task description

### What Gets Recorded

Each frame contains:
| Key | Description | Shape |
|-----|-------------|-------|
| `observation.images.front` | Camera RGB frame | (480, 640, 3) |
| `observation.state` | Follower joint positions | (6,) |
| `action` | Target joint positions | (6,) |
| `task` | Language instruction | string |

## Requirements

- Python 3.10+
- SO101 leader + follower arms
- USB camera (webcam works)
- ~50 demonstration episodes for fine-tuning

## Setup

```bash
# Create virtual environment
uv venv
.venv\Scripts\activate  # Windows

# Install PyTorch with CUDA
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install lerobot with SmolVLA support
uv pip install "lerobot[smolvla]"
```

## Step 1: Find USB Ports

```bash
python -m lerobot.find_port
```

Note the ports for your leader and follower arms (e.g., `COM3`, `COM4` on Windows).

## Step 2: Calibrate Arms (One-Time)

```bash
# Calibrate follower
python -m lerobot.calibrate \
  --robot.type=so101_follower \
  --robot.port=COM3 \
  --robot.id=my_follower

# Calibrate leader
python -m lerobot.calibrate \
  --robot.type=so101_leader \
  --robot.port=COM4 \
  --robot.id=my_leader
```

Follow the prompts to move joints through their range of motion.

## Step 3: Test Teleoperation

```bash
python -m lerobot.teleoperate \
  --robot.type=so101_follower \
  --robot.port=COM3 \
  --robot.id=my_follower \
  --teleop.type=so101_leader \
  --teleop.port=COM4 \
  --teleop.id=my_leader
```

Move the leader arm - the follower should mirror it.

## Step 4: Collect Data

```bash
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=COM3 \
  --robot.id=my_follower \
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
  --teleop.type=so101_leader \
  --teleop.port=COM4 \
  --teleop.id=my_leader \
  --display_data=true \
  --dataset.repo_id=YOUR_USERNAME/my_dataset \
  --dataset.num_episodes=50 \
  --dataset.single_task="pick up the red block"
```

Or use the helper script:
```bash
python collect_data.py
```

### Tips for Good Data

- **Consistency**: Start objects in similar positions each episode
- **Variety**: Vary positions slightly to help generalization
- **Clean demos**: Smooth movements, no mistakes
- **50+ episodes**: More data = better performance

## Step 5: Fine-tune SmolVLA

```bash
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=YOUR_USERNAME/my_dataset \
  --batch_size=64 \
  --steps=20000 \
  --output_dir=outputs/smolvla_finetuned
```

Training time: ~4 hours on A100, longer on consumer GPUs.

## Step 6: Deploy Policy

```bash
lerobot-eval \
  --policy.path=outputs/smolvla_finetuned \
  --robot.type=so101_follower \
  --robot.port=COM3 \
  --robot.id=my_follower
```

## Files

| File | Description |
|------|-------------|
| `collect_data.py` | Data collection helper with CLI commands |
| `test_smolvla_simple.py` | Test inference on your hardware |
| `requirements.txt` | Python dependencies |

## Model Info

| Property | Value |
|----------|-------|
| Base Model | `lerobot/smolvla_base` |
| Parameters | 450M |
| Architecture | SmolVLM2 backbone + Action Expert |
| Recommended Episodes | 50+ |
| Fine-tune Steps | 20,000 |

## References

- [SmolVLA Docs](https://huggingface.co/docs/lerobot/en/smolvla)
- [LeRobot Getting Started](https://huggingface.co/docs/lerobot/en/getting_started_real_world_robot)
- [SmolVLA Blog](https://huggingface.co/blog/smolvla)
- [SmolVLA Paper](https://arxiv.org/abs/2506.01844)
- [SO100/SO101 Wiki](https://wiki.seeedstudio.com/lerobot_so100m/)
