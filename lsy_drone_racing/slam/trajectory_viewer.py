import cv2
import numpy as np


class TrajectoryViewer:
    def __init__(self, window_name="Drone Trajectory", scale=10):
        self.window_name = window_name
        self.scale = scale
        # Create a blank black canvas (800x800 pixels)
        self.canvas = np.zeros((800, 800, 3), dtype=np.uint8)
        # Start drawing from the center of the image
        self.center_x = 400
        self.center_y = 400

    def update(self, pose_matrix):
        # Only draw if we have a valid 4x4 pose matrix
        if pose_matrix is None or not isinstance(pose_matrix, np.ndarray):
            return

        # Extract the translation vector (x, y, z) from the 4x4 matrix
        # In standard camera coordinates: x is right, y is down, z is forward
        x = pose_matrix[0, 3]
        z = pose_matrix[2, 3]

        # Scale the coordinates and shift them to the center of the canvas
        draw_x = int(x * self.scale) + self.center_x
        draw_y = int(z * self.scale) + self.center_y

        # Draw a green dot for the current position
        cv2.circle(self.canvas, (draw_x, draw_y), 2, (0, 255, 0), -1)

        # Show the updated map alongside your drone's camera feed
        cv2.imshow(self.window_name, self.canvas)
        cv2.waitKey(1)
