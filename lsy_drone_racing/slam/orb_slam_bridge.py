import cv2 as cv
import zmq
import time
import struct


class OrbSlamBridge:

    def __init__(self):

        context = zmq.Context()
        self.zmq_socket = context.socket(zmq.PUB)

        # "127.0.0.1" is your own computer (localhost)
        self.zmq_socket.bind("tcp://127.0.0.1:5555")

    def publish_frame(self, frame, curr_time=None):

        if frame is None:
            return

        # print(type(frame))
        # exit()
        success, encoded_image = cv.imencode(".jpg", frame)

        if not success:
            return

        image_bytes = encoded_image.tobytes()

        time_bytes = struct.pack("d", time.time())

        self.zmq_socket.send_multipart([time_bytes, image_bytes])
