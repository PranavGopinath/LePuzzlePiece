"""
Find available cameras using OpenCV.

This script will test camera indices from 0 to 9 and report which ones are available.
"""

import cv2
import sys

def find_cameras(max_index=10):
    """Test camera indices and return list of available cameras."""
    available = []
    
    print("Scanning for cameras...")
    print("-" * 60)
    
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # Try to read a frame to confirm it's working
            ret, frame = cap.read()
            if ret:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                print(f"✓ Camera {i}: Available ({width}x{height}, {fps:.1f} fps)")
                available.append(i)
            else:
                print(f"✗ Camera {i}: Opened but cannot read frames")
            cap.release()
        else:
            print(f"✗ Camera {i}: Not available")
    
    print("-" * 60)
    
    if available:
        print(f"\nFound {len(available)} available camera(s): {available}")
        print(f"\nUse camera index {available[0]} in your scripts (e.g., --camera {available[0]})")
    else:
        print("\nNo cameras found. Make sure your camera is connected and not in use by another application.")
    
    return available

if __name__ == "__main__":
    max_index = 10
    if len(sys.argv) > 1:
        try:
            max_index = int(sys.argv[1])
        except ValueError:
            print(f"Invalid max_index: {sys.argv[1]}, using default: 10")
    
    find_cameras(max_index)

