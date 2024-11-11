import numpy as np
from PIL import Image
from libcom import PainterlyHarmonizationModel
from libcom.utils.process_image import make_image_grid


class PainterlyImageHarmonizer:
    def __init__(self, device: int = 0, model_type: str = 'PHDNet'):
        self.model = PainterlyHarmonizationModel(device=device, model_type=model_type)

    def harmonize(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        # Convert PIL images to numpy arrays
        image_np = np.array(image.convert('RGB'))
        mask_np = np.array(mask.convert('L'))  # Convert mask to grayscale

        # Perform harmonization
        output_np = self.model(image_np, mask_np)

        # Convert the output back to a PIL image
        output_image = Image.fromarray(output_np)
        return output_image

    def display_result(self, image: Image.Image, mask: Image.Image):
        output_image = self.harmonize(image, mask)

        # Create a grid of images for visualization
        grid_img = make_image_grid([np.array(image), np.array(mask.convert('RGB')), np.array(output_image)])

        # Convert grid to a PIL image and display
        grid_pil = Image.fromarray(grid_img)
        grid_pil.show()