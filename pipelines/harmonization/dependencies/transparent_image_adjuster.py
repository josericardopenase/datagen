from typing import Optional
from PIL import Image

class TransparentImageAdjuster:
    def __init__(self):
        ...
    def adjust(self, image: Image.Image, threshold: float = 0) -> Image.Image:
        if image.mode != "RGBA":
            raise ValueError("Image must be in RGBA format")
        width, height = image.size
        pixels = image.load()
        left, right, top, bottom = width, 0, height, 0
        for y in range(height):
            for x in range(width):
                _, _, _, alpha = pixels[x, y]
                if alpha > threshold:
                    left = min(left, x)
                    right = max(right, x)
                    top = min(top, y)
                    bottom = max(bottom, y)
        if left > right or top > bottom:
            return Image.new("RGBA", (1, 1), (0, 0, 0, 0))
        cropped_image = image.crop((left, top, right + 1, bottom + 1))
        return cropped_image
