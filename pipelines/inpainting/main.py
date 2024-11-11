from typing import Tuple
from PIL import Image

from pipelines.inpainting.dependencies.segmentation_mask_generator import SegmentationMaskGenerator
from pipelines.inpainting.dependencies.image_cropper import ImageCropper
from pipelines.inpainting.dependencies.image_inpainter import StableDiffusionImageInpainter, ImageInpainter
from pipelines.inpainting.dependencies.image_paster import ImagePaster
from pipelines.inpainting.dependencies.mask_creator import MaskCreator
from pipelines.utils import plot_images, draw_square_inside_image
import numpy as np
import sys

class InpaintingDatasetGenerator:
    def __init__(self,
                 inpainter: ImageInpainter,
                 mask_creator: MaskCreator,
                 image_cropper : ImageCropper,
                 image_paster : ImagePaster,
                 segmentation_mask_generator : SegmentationMaskGenerator
                 ):
        self.inpainter = inpainter
        self.mask_creator = mask_creator
        self.image_cropper = image_cropper
        self.image_paster = image_paster
        self.segmentation_mask_generator = segmentation_mask_generator


    def generate(self,  image : Image.Image, resolution: Tuple[int, int], save_as="result1"):
        point = (450, 450)
        cropped_image = self.image_cropper.crop(
            image=image,
            center=point,
            resolution=resolution)
        mask = self.mask_creator.create((resolution[0] // 2, resolution[1] // 2), resolution, size_of_shape=0.15)
        inpaint = self.inpainter.inpaint(
            prompt="a boat crossing the sea",
            original_image=cropped_image,
            mask_image=mask
        )
        image_paster = ImagePaster()
        pasted = image_paster.paste(original_image=image, pasted_image=inpaint, center=point)
        segmentation_mask = self.segmentation_mask_generator.generate(image, pasted)

        plot_images(
            [
                image,
                draw_square_inside_image(image, cropped_image.size, point, border_width=7, center_radius=10),
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
image = Image.open("assets/bgs/bg.jpg")

for iteration in range(0, 10):
    dataset_generator = InpaintingDatasetGenerator(
        inpainter=StableDiffusionImageInpainter(),
        mask_creator=MaskCreator(
            shape=Image.open("assets/masks/square_mask.png")
        ),
        image_cropper=ImageCropper(),
        image_paster=ImagePaster(),
        segmentation_mask_generator=SegmentationMaskGenerator(0.995, 8)
    )
    dataset_generator.generate(
        image=image,
        resolution=(512, 512),
        save_as='result/result_{}.png'.format( iteration))