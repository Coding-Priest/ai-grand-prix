import cv2
import orbslam3
import numpy as np
import time


DEFAULT_VOCAB_PATH = "/home/homefree/Development/anduril-drone-race/ai-grand-prix/lsy_drone_racing/slam/ORBvoc.txt"
DEFAULT_SETTINGS_PATH = "/home/homefree/Development/anduril-drone-race/ai-grand-prix/lsy_drone_racing/slam/EuRoC.yaml"


class OrbSLAM:
    def __init__(self, vocab_path=None, settings_path=None, use_viewer=True):
        if vocab_path is None:
            vocab_path = DEFAULT_VOCAB_PATH

        if settings_path is None:
            settings_path = DEFAULT_SETTINGS_PATH

        # Initialize the SLAM system.
        # The 4th argument (use_viewer) enables the built-in Pangolin viewer.
        self.slam = orbslam3.system(
            vocab_path, settings_path, orbslam3.Sensor.MONOCULAR, True
        )

        print(dir(orbslam3))
        print(dir(orbslam3.system))
        print(dir(self.slam))
        # self.slam.set_use_viewer(use_viewer)
        self.slam.initialize()

    def update(self, image, timestamp) -> np.ndarray:
        # Fixed: using self.slam and the passed 'image' argument
        pose = self.slam.process_image_mono(image, timestamp)

        # We can move the print statement out of here to keep the loop clean,
        # but returning the pose is essential if you want to use it elsewhere.
        return pose

    def get_state(self):
        # Return the actual state enum from the system
        return self.slam.get_tracking_state()

    def shutdown(self):
        # CRITICAL: Always shut down the system.
        # This stops the mapping/loop closing threads and saves trajectories.
        self.slam.shutdown()


if __name__ == "__main__":
    from trajectory_viewer import TrajectoryViewer

    viewer = TrajectoryViewer(scale=20)
    # Initialize the wrapper
    orb = OrbSLAM(use_viewer=True)

    # Path to your video file
    video_path = (
        "/home/homefree/Development/anduril-drone-race/ai-grand-prix/drone_fly.mp4"
    )
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        exit()

    # Get the frames per second (FPS) to calculate accurate timestamps
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 30.0  # Fallback
    frame_time = 1.0 / fps
    current_timestamp = 0.0

    print(f"Starting video processing at {fps} FPS...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video stream reached.")
                break

            # ORB-SLAM expects images sequentially.
            pose = orb.update(frame, current_timestamp)
            state = orb.get_state()

            # Optional console tracking
            if state == orbslam3.TrackingState.LOST:
                print(f"Tracking LOST at {current_timestamp:.2f}s")
            elif state == orbslam3.TrackingState.OK:
                viewer.update(pose)  # Running smoothly

            # Advance the timestamp
            current_timestamp += frame_time

            # Sleep briefly to ensure the Pangolin viewer processes
            # and renders at roughly normal playback speed
            time.sleep(frame_time)

    except KeyboardInterrupt:
        print("\nProcessing interrupted by user.")

    finally:
        print("Releasing resources and shutting down SLAM...")
        cap.release()
        orb.shutdown()
