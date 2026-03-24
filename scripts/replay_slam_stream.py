#!/usr/bin/env python3
import argparse
import socket
import struct
import time

def replay_stream(record_path: str, host: str, port: int):
    # Connect to the SLAM server
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((host, port))
        print(f"Connected to SLAM server at {host}:{port}")
    except ConnectionRefusedError:
        print(f"Failed to connect to SLAM server at {host}:{port}. Ensure it is running.")
        return

    start_time_real = time.monotonic()
    start_timestamp_stream = None

    print(f"Replaying {record_path}...")
    try:
        with open(record_path, "rb") as f:
            while True:
                # Read the 8-byte header: [type (uint32)] [size (uint32)]
                header = f.read(8)
                if not header or len(header) < 8:
                    break
                    
                msg_type, payload_size = struct.unpack("<II", header)
                
                # Identify if we need to read a payload
                payload = b""
                if payload_size > 0:
                    payload = f.read(payload_size)
                    if len(payload) < payload_size:
                        print("Warning: Unexpected end of file matching payload size.")
                        break
                
                # Extract the timestamp if this is a Frame (1) or IMU (2) message
                if msg_type in (1, 2) and payload_size >= 8:
                    timestamp = struct.unpack("<d", payload[:8])[0]
                    
                    if start_timestamp_stream is None:
                        start_timestamp_stream = timestamp
                        start_time_real = time.monotonic()
                    else:
                        # Calculate how long we should wait to match the recorded timing
                        target_time = start_time_real + (timestamp - start_timestamp_stream)
                        now = time.monotonic()
                        if target_time > now:
                            time.sleep(target_time - now)

                # Send the exact binary copy over the socket
                sock.sendall(header)
                if payload:
                    sock.sendall(payload)
                    
                # Message 0 usually means end of stream
                if msg_type == 0:
                    break
                    
    except KeyboardInterrupt:
        print("\nReplay interrupted by user.")
    finally:
        sock.close()
        print("Replay finished and socket closed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay a binary SLAM stream recording.")
    parser.add_argument("record_path", help="Path to the binary recording file (.bin).")
    parser.add_argument("--host", default="localhost", help="SLAM server host (default: localhost).")
    parser.add_argument("--port", type=int, default=9999, help="SLAM server port (default: 9999).")
    args = parser.parse_args()
    
    replay_stream(args.record_path, args.host, args.port)
