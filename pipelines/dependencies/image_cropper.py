from PIL import Image
from typing import Tuple

class ImageCropper:
    def __init__(self):
        ...

    def crop(self, image: Image.Image, center: Tuple[int, int], resolution: Tuple[int, int]) -> Image.Image:
        cx, cy = center
        width, height = resolution
        left = max(cx - width // 2, 0)
        top = max(cy - height // 2, 0)
        right = min(cx + width // 2, image.width)
        bottom = min(cy + height // 2, image.height)
        return image.crop((left, top, right, bottom))
