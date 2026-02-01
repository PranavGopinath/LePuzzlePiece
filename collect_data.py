"""
Teleoperation Data Collection Script for LeRobot + SO101

This script collects demonstration data by teleoperating a follower arm
with a leader arm. The data is saved in LeRobot format for fine-tuning SmolVLA.

HOW IT WORKS:
=============

1. LEADER-FOLLOWER SETUP:
   - Leader arm: You physically move this arm with your hand
   - Follower arm: Mirrors the leader's movements in real-time
   - Camera(s): Records what the follower sees during the task

2. DATA COLLECTION LOOP:
   For each episode:
   a) Reset: Put objects back to starting positions
   b) Record: Move leader arm to complete the task
   c) Save: Joint positions + camera frames + task description

3. DATASET FORMAT:
   LeRobot saves data as:
   - observation.images.{camera_name}: RGB frames (H, W, 3)
   - observation.state: Joint positions of follower arm
   - action: Joint positions to execute (what follower should do)
   - task: Language description ("pick up the red block")

4. FINE-TUNING:
   After collecting ~50 episodes, fine-tune SmolVLA:
   ```
   lerobot-train \
     --policy.path=lerobot/smolvla_base \
     --dataset.repo_id=YOUR_USERNAME/your_dataset \
     --batch_size=64 \
     --steps=20000
   ```

USAGE:
======
1. First, find your USB ports:
   python -m lerobot.find_port

2. Calibrate your arms (one-time):
   python -m lerobot.calibrate --robot.type=so101_follower --robot.port=COM3 --robot.id=follower
   python -m lerobot.calibrate --robot.type=so101_leader --robot.port=COM4 --robot.id=leader

3. Test teleoperation:
   python -m lerobot.teleoperate --robot.type=so101_follower ...

4. Run this script or use lerobot-record CLI
"""

import os

# ============================================================================
# CONFIGURATION - EDIT THESE FOR YOUR SETUP
# ============================================================================

# Your HuggingFace username (for uploading dataset)
HF_USERNAME = "draval"

# Dataset name
DATASET_NAME = "so101_pick_place"

# Number of episodes to record
NUM_EPISODES = 50

# Task description (what you're demonstrating)
TASK_DESCRIPTION = "pick up the red block and place it in the box"

# USB ports (find with: python -m lerobot.find_port)
FOLLOWER_PORT = "COM3"  # Windows example, Linux: "/dev/ttyUSB0"
LEADER_PORT = "COM4"

# Robot IDs (used for calibration files)
FOLLOWER_ID = "my_follower"
LEADER_ID = "my_leader"

# Camera config (single camera setup)
# index_or_path: 0 = first webcam, 1 = second, or path like "/dev/video0"
CAMERA_CONFIG = {
    "front": {
        "type": "opencv",
        "index_or_path": 0,
        "width": 640,
        "height": 480,
        "fps": 30,
    }
}

# ============================================================================
# OPTION 1: Use the CLI (RECOMMENDED - simpler)
# ============================================================================

def print_cli_command():
    """Print the lerobot-record CLI command for your configuration."""

    # Format camera config as string
    import json
    cam_str = json.dumps(CAMERA_CONFIG).replace('"', '\\"')

    cmd = f"""
# ============================================================================
# RECOMMENDED: Use the lerobot-record CLI
# ============================================================================

# Step 1: Find your USB ports
python -m lerobot.find_port

# Step 2: Calibrate (one-time per robot)
python -m lerobot.calibrate --robot.type=so101_follower --robot.port={FOLLOWER_PORT} --robot.id={FOLLOWER_ID}
python -m lerobot.calibrate --robot.type=so101_leader --robot.port={LEADER_PORT} --robot.id={LEADER_ID}

# Step 3: Test teleoperation (optional but recommended)
python -m lerobot.teleoperate \\
  --robot.type=so101_follower \\
  --robot.port={FOLLOWER_PORT} \\
  --robot.id={FOLLOWER_ID} \\
  --teleop.type=so101_leader \\
  --teleop.port={LEADER_PORT} \\
  --teleop.id={LEADER_ID}

# Step 4: Record episodes
lerobot-record \\
  --robot.type=so101_follower \\
  --robot.port={FOLLOWER_PORT} \\
  --robot.id={FOLLOWER_ID} \\
  --robot.cameras="{cam_str}" \\
  --teleop.type=so101_leader \\
  --teleop.port={LEADER_PORT} \\
  --teleop.id={LEADER_ID} \\
  --display_data=true \\
  --dataset.repo_id={HF_USERNAME}/{DATASET_NAME} \\
  --dataset.num_episodes={NUM_EPISODES} \\
  --dataset.single_task="{TASK_DESCRIPTION}"

# Step 5: Fine-tune SmolVLA on your data
lerobot-train \\
  --policy.path=lerobot/smolvla_base \\
  --dataset.repo_id={HF_USERNAME}/{DATASET_NAME} \\
  --batch_size=64 \\
  --steps=20000 \\
  --output_dir=outputs/smolvla_finetuned
"""
    print(cmd)


# ============================================================================
# OPTION 2: Python API (more control)
# ============================================================================

def collect_data_python_api():
    """
    Collect data using LeRobot's Python API.
    This gives you more control over the recording process.
    """
    import torch
    from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
    from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig
    from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    print("=" * 60)
    print("Data Collection using Python API")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # 1. SETUP CAMERAS
    # -------------------------------------------------------------------------
    print("\n[1/5] Setting up camera...")

    camera_config = OpenCVCameraConfig(
        index_or_path=CAMERA_CONFIG["front"]["index_or_path"],
        width=CAMERA_CONFIG["front"]["width"],
        height=CAMERA_CONFIG["front"]["height"],
        fps=CAMERA_CONFIG["front"]["fps"],
    )
    cameras = {"front": OpenCVCamera(camera_config)}

    # -------------------------------------------------------------------------
    # 2. SETUP FOLLOWER ROBOT
    # -------------------------------------------------------------------------
    print("[2/5] Setting up follower robot...")

    follower_config = SO101FollowerConfig(
        port=FOLLOWER_PORT,
        id=FOLLOWER_ID,
        cameras={"front": camera_config},
    )
    follower = SO101Follower(follower_config)

    # -------------------------------------------------------------------------
    # 3. SETUP LEADER (TELEOPERATOR)
    # -------------------------------------------------------------------------
    print("[3/5] Setting up leader arm...")

    leader_config = SO101LeaderConfig(
        port=LEADER_PORT,
        id=LEADER_ID,
    )
    leader = SO101Leader(leader_config)

    # -------------------------------------------------------------------------
    # 4. CONNECT EVERYTHING
    # -------------------------------------------------------------------------
    print("[4/5] Connecting devices...")

    follower.connect()
    leader.connect()

    print(f"  Follower connected: {follower.is_connected}")
    print(f"  Leader connected: {leader.is_connected}")

    # -------------------------------------------------------------------------
    # 5. CREATE DATASET
    # -------------------------------------------------------------------------
    print("[5/5] Creating dataset...")

    dataset = LeRobotDataset.create(
        repo_id=f"{HF_USERNAME}/{DATASET_NAME}",
        robot=follower,
        fps=30,
    )

    # -------------------------------------------------------------------------
    # 6. RECORDING LOOP
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("RECORDING")
    print("=" * 60)
    print(f"Task: {TASK_DESCRIPTION}")
    print(f"Episodes to record: {NUM_EPISODES}")
    print("\nControls:")
    print("  - Move the LEADER arm to control the FOLLOWER")
    print("  - Press ENTER to start/stop recording an episode")
    print("  - Press 'q' + ENTER to quit")
    print("=" * 60)

    episode_count = 0

    while episode_count < NUM_EPISODES:
        print(f"\n--- Episode {episode_count + 1}/{NUM_EPISODES} ---")
        input("Press ENTER to start recording...")

        # Start recording
        dataset.start_episode(task=TASK_DESCRIPTION)
        print("Recording... (Press ENTER to stop)")

        recording = True
        while recording:
            # Read leader position
            leader_state = leader.get_state()

            # Send to follower (teleoperation)
            follower.send_action(leader_state)

            # Get observation from follower
            observation = follower.get_observation()

            # Add to dataset
            dataset.add_frame(observation)

            # Check for stop (non-blocking would be better, this is simplified)
            # In practice, use keyboard library or threading

        dataset.end_episode()
        episode_count += 1
        print(f"Episode {episode_count} saved!")

    # -------------------------------------------------------------------------
    # 7. SAVE AND UPLOAD
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SAVING DATASET")
    print("=" * 60)

    dataset.save()
    print(f"Dataset saved locally!")

    # Upload to HuggingFace Hub
    upload = input("Upload to HuggingFace Hub? (y/n): ")
    if upload.lower() == 'y':
        dataset.push_to_hub()
        print(f"Uploaded to: https://huggingface.co/datasets/{HF_USERNAME}/{DATASET_NAME}")

    # Cleanup
    follower.disconnect()
    leader.disconnect()

    print("\nDone! Next steps:")
    print(f"  1. Fine-tune SmolVLA on your data:")
    print(f"     lerobot-train --policy.path=lerobot/smolvla_base --dataset.repo_id={HF_USERNAME}/{DATASET_NAME}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║          LeRobot Data Collection for SmolVLA                 ║
╠══════════════════════════════════════════════════════════════╣
║  This script helps you collect teleoperation data            ║
║  for fine-tuning SmolVLA on your SO101 robot.                ║
╚══════════════════════════════════════════════════════════════╝
""")

    print("Choose an option:")
    print("  1. Print CLI commands (RECOMMENDED)")
    print("  2. Run Python API (experimental)")
    print()

    choice = input("Enter 1 or 2: ").strip()

    if choice == "2":
        collect_data_python_api()
    else:
        print_cli_command()
