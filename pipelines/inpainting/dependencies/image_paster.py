from PIL import Image
from typing import Tuple

class ImagePaster:
    def __init__(self):
        ...

    def paste(self, original_image: Image.Image, pasted_image: Image.Image, center: Tuple[int, int]) -> Image.Image:
        result_image = original_image.copy()
        cx, cy = center
        left = cx - pasted_image.width // 2
        top = cy - pasted_image.height // 2
        right = left + pasted_image.width
        bottom = top + pasted_image.height
        paste_box = (max(0, left), max(0, top), min(right, original_image.width), min(bottom, original_image.height))
        crop_box = (paste_box[0] - left, paste_box[1] - top, paste_box[2] - left, paste_box[3] - top)
        cropped_pasted_image = pasted_image.crop(crop_box)
        result_image.paste(cropped_pasted_image, paste_box)
        return result_image
