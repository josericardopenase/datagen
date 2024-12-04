from PIL import Image
from typing import Tuple

class ImageCropper:
    def __init__(self):
        ...

    def crop(self, image: Image.Image, center: Tuple[int, int], resolution: Tuple[int, int]) -> Image.Image:
        cx, cy = center
        width, height = resolution
        left = cx - width // 2
        top = cy - height // 2
        right = cx + width // 2
        bottom = cy + height // 2

        cropped_image = Image.new("RGBA", (width, height), (0, 0, 0, 0))

        paste_left = max(0, -left)
        paste_top = max(0, -top)

        src_left = max(0, left)
        src_top = max(0, top)
        src_right = min(image.width, right)
        src_bottom = min(image.height, bottom)

        paste_width = src_right - src_left
        paste_height = src_bottom - src_top

        cropped_image.paste(
            image.crop((src_left, src_top, src_right, src_bottom)),
            (paste_left, paste_top, paste_left + paste_width, paste_top + paste_height)
        )

        return cropped_image
