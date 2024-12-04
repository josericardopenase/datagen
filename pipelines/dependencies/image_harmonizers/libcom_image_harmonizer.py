import numpy as np
from PIL import Image
from libcom import ImageHarmonizationModel
from libcom.utils.process_image import make_image_grid

from pipelines.dependencies.image_harmonizers.image_harmonizer import ImageHarmonizer


class LibcomImageHarmonizer(ImageHarmonizer):
    def __init__(self, device: int = 0, model_type: str = 'CDTNet'):
        self.model = ImageHarmonizationModel(device=device, model_type=model_type)

    def harmonize(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        image_np = np.array(image.convert('RGB'))
        mask_np = np.array(mask.convert('L'))
        output_np = self.model(image_np, mask_np)
        output_image = Image.fromarray(output_np.astype(np.uint8))
        return output_image

    def display_result(self, image: Image.Image, mask: Image.Image):
        output_image = self.harmonize(image, mask)
        grid_img = make_image_grid([np.array(image), np.array(mask.convert('RGB')), np.array(output_image)])
        grid_pil = Image.fromarray(grid_img)
        grid_pil.show()