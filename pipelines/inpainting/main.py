from typing import Tuple
from PIL import Image

from pipelines.inpainting.dependencies.bounding_box_generator import SegmentationMaskGenerator
from pipelines.inpainting.dependencies.image_cropper import ImageCropper
from pipelines.inpainting.dependencies.image_inpainter import StableDiffusionImageInpainter
from pipelines.inpainting.dependencies.image_paster import ImagePaster
from pipelines.inpainting.dependencies.mask_creator import MaskCreator
from pipelines.utils import plot_images, draw_square_inside_image
import numpy as np
import sys

class InpaintingDatasetGenerator:
    def __init__(self, image : Image.Image, shape: Image.Image):
        self.image = image
        self.shape = shape

    def generate(self, resolution: Tuple[int, int], save_as="result1"):
        point = (450, 450)
        cropper = ImageCropper(self.image)
        cropped_image = cropper.crop(point, resolution)
        cropped_image.save("img1.png")
        mask_creator = MaskCreator(shape=shape)
        mask = mask_creator.create((resolution[0] // 2, resolution[1] // 2), resolution, size_of_shape=0.15)
        inpainter = StableDiffusionImageInpainter(
            prompt="a boat crossing the sea",
            original_image=cropped_image,
            mask_image=mask
        )
        inpaint = inpainter.inpaint()[0]
        inpaint.save("img2.png")
        image_paster = ImagePaster(original_image=self.image, pasted_image=inpaint)
        pasted = image_paster.paste(point)

        segmentation_mask_generator = SegmentationMaskGenerator(0.995, 8)
        segmentation_mask = segmentation_mask_generator.generate(self.image, pasted)

        plot_images(
            [
                self.image,
                draw_square_inside_image(self.image, cropped_image.size, point, border_width=7, center_radius=10),
                cropped_image,
                mask,
                (np.array(cropped_image.convert('1')) + np.array(mask.convert('1'))),
                inpaint,
                pasted,
                segmentation_mask
            ],
            ["Imágen original", "Posición de recorte", "Recorte", "Mascara generada", "Mascara aplicada", "Inpainted Image", "Finally pasted image", "Máscara de segmentación"],
            main_title="Pipeline",
            save_as=save_as
        )
        return pasted

folder = sys.argv[0] if sys.argv[0] else 0

for x in range(0, 10):
    image = Image.open("assets/bgs/bg.jpg")
    shape = Image.open("assets/masks/square_mask.png")
    dataset_generator = InpaintingDatasetGenerator(
        image=image,
        shape=shape
    )
    dataset_generator.generate((512, 512), save_as='result_{}.png'.format(x))