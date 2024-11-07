from typing import Tuple
from PIL import Image
from pipelines.inpainting.dependencies.image_cropper import ImageCropper
from pipelines.inpainting.dependencies.image_inpainter import StableDiffusionImageInpainter
from pipelines.inpainting.dependencies.image_paster import ImagePaster
from pipelines.inpainting.dependencies.mask_creator import MaskCreator
from pipelines.utils import plot_images, draw_square_inside_image
import numpy as np

class InpaintingDatasetGenerator:
    def __init__(self, image : Image.Image, shape: Image.Image):
        self.image = image
        self.shape = shape

    def generate(self, resolution: Tuple[int, int]):
        point = (450, 450)
        cropper = ImageCropper(self.image)
        cropped_image = cropper.crop(point, resolution)
        mask_creator = MaskCreator(shape=shape)
        mask = mask_creator.create((resolution[0] // 2, resolution[1] // 2), resolution, size_of_shape=0.15)
        inpainter = StableDiffusionImageInpainter(
            prompt="a boat",
            original_image=cropped_image,
            mask_image=mask
        )
        inpaint = inpainter.inpaint()[0]
        image_paster = ImagePaster(original_image=inpaint, pasted_image=mask)
        pasted = image_paster.paste(point)

        plot_images(
            [
                self.image,
                draw_square_inside_image(self.image, cropped_image.size, point, border_width=7, center_radius=10),
                cropped_image,
                mask,
                (np.array(cropped_image.convert('1')) + np.array(mask.convert('1'))),
                pasted,
            ],
            ["Imágen original", "Posición de recorte", "Recorte", "Mascara generada", "Mascara aplicada", "Finally pasted image"],
            main_title="Pipeline"
        )

image = Image.open("assets/bgs/bg.jpg")
shape = Image.open("assets/masks/square_mask.png")
dataset_generator = InpaintingDatasetGenerator(
    image=image,
    shape=shape
)
dataset_generator.generate((500, 500))