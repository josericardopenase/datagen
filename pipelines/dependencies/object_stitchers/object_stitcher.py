from abc import ABC, abstractmethod
from typing import Tuple

from PIL import Image


class ObjectStitcher(ABC):
    @abstractmethod
    def stitch(self, bg : Image.Image, fg: Image.Image, fg_mask: Image.Image, bbox: Tuple[int, int, int, int]) -> Image.Image:
        ...