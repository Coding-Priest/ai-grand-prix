import cv2
import numpy as np
import pyslam
from pyslam.slam.tracking import Tracking


def run_drone_slam(video_source=0):
    # 1. Setup Configuration (use your calibrated camera file)
    # If you don't have one, pySLAM has defaults in its 'params' folder
    config = Config()

    # 2. Initialize SLAM System
    # sensor_type: MONOCULAR is standard for most drone setups
    # use_viewer: True opens the Pangolin/Rerun window automatically
    slam = System(config, sensor_type=SensorType.MONOCULAR, use_viewer=True)

    cap = cv2.VideoCapture(video_source)

    print("SLAM Initialized. Press 'q' to stop.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        # 1. Feed frame to pySLAM's Tracking module
        tracking.process_frame(frame, timestamp)

        # 2. Get the pose for your drone logic
        if tracking.state == Tracking.TrackingState.OK:
            pose = tracking.current_frame.pose  # The 4x4 matrix
            print(f"Drone Location: {pose[:3, 3]}")

    slam.shutdown()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_drone_slam()
