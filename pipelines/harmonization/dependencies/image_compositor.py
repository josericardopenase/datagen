from typing import Tuple
from PIL import Image

class ImageCompositor:
    def composite(
        self,
        background: Image.Image,
        foreground: Image.Image,
        center: Tuple[int, int],
        size_of: float = 1
    ) -> Tuple[Image.Image, Tuple[int, int]]:
        fg_width, fg_height = foreground.size
        bg_width, bg_height = background.size

        # Calculate maximum dimensions for the foreground based on size_of
        max_width = int(bg_width * size_of)
        max_height = int(bg_height * size_of)

        # Maintain aspect ratio while resizing the foreground
        aspect_ratio = fg_width / fg_height
        if max_width / aspect_ratio <= max_height:
            new_width = max_width
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = max_height
            new_width = int(new_height * aspect_ratio)

        resized_foreground = foreground.resize((new_width, new_height), resample=Image.ANTIALIAS)

        center_x, center_y = center
        top_left_x = max(0, center_x - new_width // 2)
        top_left_y = max(0, center_y - new_height // 2)
        bottom_right_x = min(bg_width, top_left_x + new_width)
        bottom_right_y = min(bg_height, top_left_y + new_height)

        # Crop the foreground to fit within the boundaries
        crop_left = max(0, -1 * (center_x - new_width // 2))
        crop_top = max(0, -1 * (center_y - new_height // 2))
        crop_right = crop_left + (bottom_right_x - top_left_x)
        crop_bottom = crop_top + (bottom_right_y - top_left_y)

        cropped_foreground = resized_foreground.crop((crop_left, crop_top, crop_right, crop_bottom))

        # Create a copy of the background to paste onto
        composite_image = background.copy()
        composite_image.paste(cropped_foreground, (top_left_x, top_left_y), cropped_foreground if cropped_foreground.mode == 'RGBA' else None)

        return composite_image, (cropped_foreground.size)
