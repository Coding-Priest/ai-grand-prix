import pyorbslam
import numpy as np


class OrbSLAM:

    def __init__(self, vocab_path=None, settings_path=None):

        default_vocab_path = "ORBvoc.txt"
        default_settings_path = "EuRoC.yaml"

        self.slam = pyorbslam.MonoSLAM(vocab_path, settings_path)

    def update(self, image) -> void:
        pass

    def get_state(self):
        pass


if __name__ == "__main__":
    orb = OrbSLAM()
