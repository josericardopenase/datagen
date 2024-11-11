from PIL import Image


class TransparentImageCleaner:
    def __init__(self, threshold: float):
        # Ensure the threshold is between 0.0 and 1.0
        if not (0.0 <= threshold <= 1.0):
            raise ValueError("Threshold must be between 0.0 and 1.0")
        self.threshold = int(threshold * 255)  # Convert threshold to 0-255 range

    def clean(self, image: Image.Image) -> Image.Image:
        if image.mode != "RGBA":
            raise ValueError("Image must be in RGBA format")

        # Create a new image to store the cleaned result
        cleaned_image = Image.new("RGBA", image.size)
        pixels = image.load()
        cleaned_pixels = cleaned_image.load()

        # Iterate over every pixel in the image
        for y in range(image.height):
            for x in range(image.width):
                r, g, b, alpha = pixels[x, y]

                # Check if the alpha value is below the threshold
                if alpha < self.threshold:
                    # Make the pixel fully transparent
                    cleaned_pixels[x, y] = (0, 0, 0, 0)
                else:
                    # Keep the original pixel
                    cleaned_pixels[x, y] = (r, g, b, alpha)

        return cleaned_image
