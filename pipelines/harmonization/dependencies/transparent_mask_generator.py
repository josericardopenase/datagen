from typing import Tuple
from PIL import Image, ImageDraw, ImageFilter, ImageChops

class TransparentMaskGenerator:
    def __init__(self, fill: bool = True, border_size: int = 4, inside_border: bool = True, centered_border: bool = False):
        self.fill = fill
        self.border_size = border_size
        self.inside_border = inside_border
        self.centered_border = centered_border

    def generate(self, image: Image.Image) -> Image.Image:
        if image.mode != "RGBA":
            raise ValueError("Image must be in RGBA format")

        # Expand the image size if the border exceeds the image size
        if self.border_size * 2 > min(image.size):
            expansion = self.border_size * 2 - min(image.size)
            new_width = image.width + expansion
            new_height = image.height + expansion
            expanded_image = Image.new("RGBA", (new_width, new_height), (0, 0, 0, 0))
            expanded_image.paste(image, (expansion // 2, expansion // 2))
            image = expanded_image

        # Create a new mask image with the same size, initialized to black (0)
        mask = Image.new("L", image.size, 0)

        # Get the alpha channel from the input image
        alpha = image.split()[3]

        # Generate a binary mask based on the alpha channel (non-transparent areas)
        binary_mask = alpha.point(lambda p: 255 if p > 0 else 0)

        if self.fill:
            # Fill the entire object with white
            mask.paste(binary_mask)
        else:
            if self.centered_border:
                # Create a centered border, half inside and half outside
                expanded_mask = binary_mask.filter(ImageFilter.MaxFilter(self.border_size))
                inner_mask = binary_mask.filter(ImageFilter.MinFilter(self.border_size))
                mask = ImageChops.subtract(expanded_mask, inner_mask)
            elif self.inside_border:
                # Generate an inner border by eroding the mask
                inner_border = binary_mask.filter(ImageFilter.MinFilter(self.border_size * 2 + 1))
                mask = ImageChops.subtract(binary_mask, inner_border)
            else:
                # Generate an outer border by expanding the mask
                expanded_mask = binary_mask.filter(ImageFilter.MaxFilter(self.border_size * 2 + 1))
                mask = ImageChops.subtract(expanded_mask, binary_mask)

        # Convert the mask to a binary (black/white) image
        final_mask = mask.point(lambda p: 255 if p > 0 else 0)

        # Return the mask as an image with mode "L" (grayscale)
        return final_mask
