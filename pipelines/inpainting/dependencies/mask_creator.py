from PIL import Image
from typing import Tuple

class MaskCreator:
    def __init__(self, shape: Image.Image, size_of_shape : float = 1):
        self.shape = shape.convert("L")
        self.size_of_shape = size_of_shape

    def create(self, center: Tuple[int, int], resolution: Tuple[int, int]) -> Image.Image:
        mask = Image.new("L", resolution, 0)
        white_bbox = self.shape.getbbox()
        if white_bbox is None:
            print("No hay regiones blancas en la forma proporcionada.")
            return mask
        cropped_shape = self.shape.crop(white_bbox)
        scaled_width = int(cropped_shape.width * self.size_of_shape)
        scaled_height = int(cropped_shape.height * self.size_of_shape)
        resized_shape = cropped_shape.resize((scaled_width, scaled_height), Image.LANCZOS)
        cx, cy = center
        left = max(cx - scaled_width // 2, 0)
        top = max(cy - scaled_height // 2, 0)
        right = min(left + scaled_width, resolution[0])
        bottom = min(top + scaled_height, resolution[1])
        resized_shape = resized_shape.crop((0, 0, right - left, bottom - top))
        mask.paste(resized_shape, (left, top))
        return mask
