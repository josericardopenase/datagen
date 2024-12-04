from typing import Tuple
from PIL import Image, ImageFilter, ImageChops

class TransparentMaskGenerator:
    def __init__(self,
                 fill: bool = True,
                 size_of: float = 0.55,
                 border_size: int = 4,
                 inside_border: bool = True,
                 centered_border: bool = False):
        self.fill = fill
        self.border_size = border_size
        self.inside_border = inside_border
        self.centered_border = centered_border
        self.size_of = size_of

    def generate(self, fg: Image.Image, resolution: Tuple[int, int]) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
        # Create a full black image with the given resolution
        mask = Image.new("L", resolution, 0)

        # Resize the foreground image according to `size_of`
        fg = fg.resize((int(fg.size[0] * self.size_of), int(fg.size[1] * self.size_of)), resample=Image.BILINEAR)
        fg_alpha = fg.split()[3]  # Use the alpha channel for transparency

        # Center the resized foreground on the black mask
        x_offset = (resolution[0] // 2) - (fg.size[0] // 2)
        y_offset = (resolution[1] // 2) - (fg.size[1] // 2)

        # Calculate bounding box
        bounding_box = (x_offset, y_offset, x_offset + fg.size[0], y_offset + fg.size[1])

        # Paste the alpha channel of the foreground onto the mask
        mask.paste(fg_alpha, (x_offset, y_offset))

        if self.fill:
            # If `fill` is True, return the fully filled mask and bounding box
            return mask, bounding_box
        else:
            # Create the border mask based on the options
            if self.inside_border:
                # Generate an inner border
                inner_border = mask.filter(ImageFilter.MinFilter(self.border_size * 2 + 1))
                border_mask = ImageChops.subtract(mask, inner_border)
            elif self.centered_border:
                # Generate a centered border
                expanded_mask = mask.filter(ImageFilter.MaxFilter(self.border_size))
                inner_mask = mask.filter(ImageFilter.MinFilter(self.border_size))
                border_mask = ImageChops.subtract(expanded_mask, inner_mask)
            else:
                # Generate an outer border
                expanded_mask = mask.filter(ImageFilter.MaxFilter(self.border_size * 2 + 1))
                border_mask = ImageChops.subtract(expanded_mask, mask)

            # Start with a full black image
            final_mask = Image.new("L", resolution, 0)

            # Paste the border mask onto the final black image
            final_mask.paste(border_mask, (0, 0))

            return final_mask, bounding_box
