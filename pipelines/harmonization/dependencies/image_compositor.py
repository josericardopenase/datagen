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

        resized_foreground = foreground.resize((new_width, new_height))

        center_x, center_y = center
        top_left_x = center_x - new_width // 2
        top_left_y = center_y - new_height // 2

        # Create a copy of the background to paste onto
        composite_image = background.copy()
        composite_image.paste(resized_foreground, (top_left_x, top_left_y), resized_foreground if resized_foreground.mode == 'RGBA' else None)

        return composite_image, (new_width, new_height)
