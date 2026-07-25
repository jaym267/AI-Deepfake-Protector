"""The four detection models.

Image / Audio / Raw Frames are independent detectors; Video Authenticator is the
meta-layer that fuses all three for video uploads. See base.py for the shared
contract and the score direction convention.
"""

from .audio_model import AudioModel
from .base import Detector, DetectorOutput
from .image_model import ImageModel
from .raw_frames_model import RawFramesModel
from .video_authenticator import VideoAuthenticator

__all__ = [
    "AudioModel",
    "Detector",
    "DetectorOutput",
    "ImageModel",
    "RawFramesModel",
    "VideoAuthenticator",
]
