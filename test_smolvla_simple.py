"""
Minimal SmolVLA test using LeRobot's evaluation framework.
"""

import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load model
print("Loading SmolVLA...")
model_id = "lerobot/smolvla_base"
policy = SmolVLAPolicy.from_pretrained(model_id)
policy.to(device)
policy.eval()
print(f"Loaded! Parameters: {sum(p.numel() for p in policy.parameters()) / 1e6:.0f}M")

# Print config to see what's available
print(f"\nConfig attributes: {[a for a in dir(policy.config) if not a.startswith('_')]}")

# Create preprocessor
preprocess, postprocess = make_pre_post_processors(
    policy.config, model_id,
    preprocessor_overrides={"device_processor": {"device": str(device)}},
)

# Get expected shapes from config
img_keys = list(policy.config.image_features.keys())
img_shape = policy.config.image_features[img_keys[0]].shape

# Get state shape from config
if hasattr(policy.config, 'robot_state_feature') and policy.config.robot_state_feature:
    state_shape = policy.config.robot_state_feature.shape
else:
    state_shape = (6,)  # Default for SO100 robot

print(f"Image keys: {img_keys}")
print(f"Image shape: {img_shape}, State shape: {state_shape}")

# Create dummy batch
# Option: Use single image duplicated across all cameras
single_image = torch.rand(1, *img_shape, device=device)
batch = {key: single_image for key in img_keys}  # Same image for all cameras

batch["observation.state"] = torch.zeros(1, *state_shape, device=device)
batch["task"] = ["pick up the red block"]

print(f"Batch keys: {list(batch.keys())}")
print(f"Using single image duplicated to {len(img_keys)} camera inputs")

# Preprocess and run inference
print("\nRunning preprocessing...")
processed = preprocess(batch)
print(f"Processed keys: {list(processed.keys())}")

print("\nRunning inference...")
with torch.inference_mode():
    action = policy.select_action(processed)

action = postprocess(action)

# Show result
print(f"\nAction type: {type(action)}")
if hasattr(action, 'shape'):
    print(f"Action shape: {action.shape}")
print(f"Action output: {action}")

if torch.cuda.is_available():
    print(f"\nGPU memory used: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

print("\nDone!")
