from PIL import Image
from typing import Tuple

class ImagePaster:
    def __init__(self, original_image: Image.Image, pasted_image: Image.Image):
        self.original_image = original_image
        self.pasted_image = pasted_image

    def paste(self, center: Tuple[int, int]) -> Image.Image:
        result_image = self.original_image.copy()
        cx, cy = center
        left = cx - self.pasted_image.width // 2
        top = cy - self.pasted_image.height // 2
        right = left + self.pasted_image.width
        bottom = top + self.pasted_image.height
        paste_box = (max(0, left), max(0, top), min(right, self.original_image.width), min(bottom, self.original_image.height))
        crop_box = (paste_box[0] - left, paste_box[1] - top, paste_box[2] - left, paste_box[3] - top)
        cropped_pasted_image = self.pasted_image.crop(crop_box)
        result_image.paste(cropped_pasted_image, paste_box)
        return result_image
