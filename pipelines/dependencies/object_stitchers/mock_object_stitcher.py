from abc import ABC, abstractmethod
from typing import Tuple

from PIL import Image

from pipelines.dependencies.object_stitchers.object_stitcher import ObjectStitcher


class MockObjectStitcher(ObjectStitcher):
    def stitch(self, bg : Image.Image, fg: Image.Image, fg_mask: Image.Image, bbox: Tuple[int, int, int, int]) -> Image.Image:
        return bg
