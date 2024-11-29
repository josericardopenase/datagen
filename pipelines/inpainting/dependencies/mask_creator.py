from PIL import Image
from typing import Tuple

class MaskCreator:
    def __init__(self, shape: Image.Image, resolution_of_shape: Tuple[float, float] = (1.0, 1.0)):
        self.shape = shape.convert("L") if shape else None
        self.resolution_of_shape = resolution_of_shape

    def create(self, center: Tuple[int, int], resolution: Tuple[int, int]) -> Image.Image:
        mask = Image.new("L", resolution, 0)
        if self.shape:
            white_bbox = self.shape.getbbox()
            if white_bbox is None:
                print("No hay regiones blancas en la forma proporcionada.")
                return mask
            cropped_shape = self.shape.crop(white_bbox)
            w_ratio, h_ratio = self.resolution_of_shape
            scaled_width = int(cropped_shape.width * w_ratio)
            scaled_height = int(cropped_shape.height * h_ratio)
            resized_shape = cropped_shape.resize((scaled_width, scaled_height), Image.LANCZOS)
        else:
            print("No se proporcionó ninguna forma. Se usará un rectángulo predeterminado.")
            w_ratio, h_ratio = self.resolution_of_shape
            scaled_width = int(resolution[0] * w_ratio)
            scaled_height = int(resolution[1] * h_ratio)
            resized_shape = Image.new("L", (scaled_width, scaled_height), 255)

        cx, cy = center
        left = max(cx - scaled_width // 2, 0)
        top = max(cy - scaled_height // 2, 0)
        right = min(left + scaled_width, resolution[0])
        bottom = min(top + scaled_height, resolution[1])
        resized_shape = resized_shape.crop((0, 0, right - left, bottom - top))
        mask.paste(resized_shape, (left, top))
        return mask
