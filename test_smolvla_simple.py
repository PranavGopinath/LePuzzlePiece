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

# Get state shape from config (try different attribute names)
if hasattr(policy.config, 'input_shapes'):
    state_shape = policy.config.input_shapes.get("observation.state", (6,))
elif hasattr(policy.config, 'robot_state_feature'):
    state_shape = policy.config.robot_state_feature.shape
elif hasattr(policy.config, 'state_feature'):
    state_shape = policy.config.state_feature.shape
else:
    # Default for SO100 robot (6 joints)
    state_shape = (6,)

print(f"Image keys: {img_keys}")
print(f"Image shape: {img_shape}, State shape: {state_shape}")

# Create dummy batch
batch = {key: torch.rand(1, *img_shape, device=device) for key in img_keys}
batch["observation.state"] = torch.zeros(1, *state_shape, device=device)
batch["task"] = ["pick up the red block"]

print(f"Batch keys: {list(batch.keys())}")

# Preprocess and run inference
print("Running inference...")
processed = preprocess(batch)
print(f"Processed keys: {list(processed.keys())}")

with torch.inference_mode():
    action = policy.select_action(processed)

action = postprocess(action)

# Show result
print(f"\nAction output: {action}")
print("\nDone!")
