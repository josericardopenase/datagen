from PIL import Image
from typing import Tuple

class ImageCropper:
    def __init__(self, image: Image.Image):
        self.image = image

    def crop(self, center: Tuple[int, int], resolution: Tuple[int, int]) -> Image.Image:
        cx, cy = center
        width, height = resolution
        left = max(cx - width // 2, 0)
        top = max(cy - height // 2, 0)
        right = min(cx + width // 2, self.image.width)
        bottom = min(cy + height // 2, self.image.height)
        return self.image.crop((left, top, right, bottom))
