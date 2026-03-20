import argparse
import socket
import struct
import time
import sys
from typing import Optional
import cv2
import numpy as np


class SLAMStreamer:
    """Streams video frames to an ORB-SLAM3 TCP server."""

    def __init__(
        self, host: str = "localhost", port: int = 9999, jpeg_quality: int = 80
    ):
        self.host = host
        self.port = port
        self.jpeg_quality = jpeg_quality
        self.sock = None
        self.frame_count = 0
        self.start_time = None

    def connect(self):
        """Connect to the SLAM TCP server."""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.host, self.port))
        self.start_time = time.monotonic()
        print(f"Connected to SLAM server at {self.host}:{self.port}")

    def send_frame(self, frame: np.ndarray, timestamp: Optional[float] = None):
        """
        Send a single frame to the SLAM server.

        Args:
            frame: BGR image as numpy array (H, W, 3), uint8.
            timestamp: Frame timestamp in seconds. If None, uses
                       elapsed time since connect().
        """
        sock = self.sock
        start_time = self.start_time
        if sock is None or start_time is None:
            raise RuntimeError("Not connected. Call connect() first.")

        if timestamp is None:
            timestamp = time.monotonic() - start_time

        # Encode frame as JPEG
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        success, jpeg_data = cv2.imencode(".jpg", frame, encode_params)
        if not success:
            raise RuntimeError("Failed to encode frame as JPEG")

        jpeg_bytes = jpeg_data.tobytes()
        payload_size = 8 + len(jpeg_bytes)  # 8 bytes for timestamp double

        # Message Type 1: Image Frame
        # Header: [message_type (uint32 LE)][payload_size (uint32 LE)][timestamp (double LE)]
        header = struct.pack("<IId", 1, payload_size, timestamp)
        sock.sendall(header)

        # Send JPEG payload
        sock.sendall(jpeg_bytes)

        self.frame_count += 1

    def send_imu(self, accel: tuple, gyro: tuple, timestamp: Optional[float] = None):
        """
        Send an IMU measurement to the SLAM server.

        Args:
            accel: (x, y, z) linear acceleration in m/s^2.
            gyro: (x, y, z) angular velocity in rad/s.
            timestamp: Measurement timestamp in seconds. Uses elapsed time if None.
        """
        sock = self.sock
        start_time = self.start_time
        if sock is None or start_time is None:
            raise RuntimeError("Not connected. Call connect() first.")

        if timestamp is None:
            timestamp = time.monotonic() - start_time

        # Message Type 2: IMU Measurement
        # Payload: [timestamp double][accel_x float][accel_y float][accel_z float]
        #          [gyro_x float][gyro_y float][gyro_z float]
        # Size = 8 (double) + 6 * 4 (floats) = 32 bytes
        payload_size = 32
        header = struct.pack("<II", 2, payload_size)
        payload = struct.pack(
            "<dffffff",
            timestamp,
            accel[0],
            accel[1],
            accel[2],
            gyro[0],
            gyro[1],
            gyro[2],
        )
        sock.sendall(header + payload)

    def send_reset(self):
        """Send a command to reset the active SLAM map."""
        sock = self.sock
        if sock is None:
            raise RuntimeError("Not connected. Call connect() first.")

        # Message Type 3: Reset Command
        # Payload size = 0
        header = struct.pack("<II", 3, 0)
        sock.sendall(header)

    def send_end_of_stream(self):
        """Signal end-of-stream to the server (message_type = 0)."""
        sock = self.sock
        if sock is not None:
            sock.sendall(struct.pack("<II", 0, 0))

    def close(self):
        """Send end-of-stream and close connection."""
        self.send_end_of_stream()
        if self.sock is not None:
            self.sock.close()
            self.sock = None
        print(f"Connection closed. Sent {self.frame_count} frames.")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.close()
