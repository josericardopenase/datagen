from typing import Tuple
from PIL import Image

from pipelines.dependencies.image_cropper import ImageCropper
from pipelines.dependencies.image_harmonizers.image_harmonizer import ImageHarmonizer
from pipelines.dependencies.image_harmonizers.mock_image_harnonizer import MockImageHarmonizer
from pipelines.dependencies.image_inpainters.image_inpainter import ImageInpainter
from pipelines.dependencies.image_inpainters.mock_image_inpainter import MockImageInpainter
from pipelines.dependencies.image_paster import ImagePaster
from pipelines.harmonization.dependencies import transparent_image_cleaner
from pipelines.harmonization.dependencies.image_compositor import ImageCompositor
from pipelines.harmonization.dependencies.transparent_image_adjuster import TransparentImageAdjuster
from pipelines.harmonization.dependencies.transparent_image_cleaner import TransparentImageCleaner
from pipelines.harmonization.dependencies.transparent_mask_generator import TransparentMaskGenerator
from pipelines.utils import plot_images, draw_square_inside_image
import sys

class HarmonizationDatasetGenerator:
    def __init__(self,
                 image_cropper : ImageCropper,
                 image_paster : ImagePaster,
                 image_compositor : ImageCompositor,
                 image_shape_adjuster : TransparentImageAdjuster,
                 harmonization_mask_generator : TransparentMaskGenerator,
                 inpainting_mask_generator : TransparentMaskGenerator,
                 transparent_image_cleaner : TransparentImageCleaner,
                 inpainter : ImageInpainter,
                 harmonizer: ImageHarmonizer
                 ):
        self.image_cropper = image_cropper
        self.image_paster = image_paster
        self.image_compositor = image_compositor
        self.image_shape_adjuster = image_shape_adjuster
        self.harmonization_mask_generator = harmonization_mask_generator
        self.inpainting_mask_generator = inpainting_mask_generator
        self.transparent_image_cleaner = transparent_image_cleaner
        self.inpainter = inpainter
        self.harmonizer = harmonizer


    def generate(self,  image : Image.Image, resolution: Tuple[int, int], save_as="result1"):
        point_of_crop = (450, 450)
        png_boat = Image.open("assets/boats/boat.png")
        cropped_image = self.image_cropper.crop(
            image=image,
            center=point_of_crop,
            resolution=resolution)
        adjusted_boat = self.image_shape_adjuster.adjust(png_boat, 1)
        cleaned_boat = self.transparent_image_cleaner.clean(adjusted_boat)
        composited_image, fg_shape = self.image_compositor.composite(
            background=cropped_image,
            foreground=cleaned_boat,
            center=(cropped_image.size[0]//2, cropped_image.size[1]//2),
            size_of=0.4
        )

        harmonization_mask = self.generate_harmonization_mask(cleaned_boat, cropped_image)
        harmonized_image = self.harmonizer.harmonize(composited_image, harmonization_mask)
        inpainting_mask, fg_shape = self.generate_inpainting_mask(cleaned_boat, cropped_image, fg_shape)
        inpainted_image = self.inpainter.inpaint(harmonized_image, inpainting_mask)
        pasted = self.image_paster.paste(
            original_image=image,
            pasted_image=inpainted_image,
            center=point_of_crop
        )
        plot_images(
            [
                image,
                draw_square_inside_image(image, cropped_image.size, point_of_crop, border_width=7, center_radius=10),
                cropped_image,
                composited_image,
                harmonization_mask,
                harmonized_image,
                inpainting_mask,
                inpainted_image,
                pasted,
                draw_square_inside_image(pasted, fg_shape, point_of_crop, border_width=7, center_radius=10)
            ],
            ["Imágen original", "Posición de recorte", "Recorte", "Barco incluido", "Mascara de harmonización","Imágen harmonizada",  "Máscara de inpainting", "Imagen con inpainting realizado", "Imágen original con región copiada", "Bounding box añadida"],
            main_title="Pipeline",
            save_as=save_as
        )
        return pasted

    def generate_inpainting_mask(self, cleaned_boat, cropped_image, fg_shape):
        composited_inpainting_mask, fg_shape = self.image_compositor.composite(
            background=Image.new("RGB", cropped_image.size, color=(0, 0, 0)),
            foreground=self.inpainting_mask_generator.generate(cleaned_boat),
            center=(cropped_image.size[0] // 2, cropped_image.size[1] // 2),
            size_of=0.4
        )
        return composited_inpainting_mask, fg_shape

    def generate_harmonization_mask(self, cleaned_boat, cropped_image):
        composited_harmonization_mask, fg_shape = self.image_compositor.composite(
            background=Image.new("RGB", cropped_image.size, color=(0, 0, 0)),
            foreground=self.harmonization_mask_generator.generate(cleaned_boat),
            center=(cropped_image.size[0] // 2, cropped_image.size[1] // 2),
            size_of=0.4
        )
        return composited_harmonization_mask


folder = sys.argv[0] if sys.argv[0] else 0
image = Image.open("assets/bgs/bg.jpg")

for iteration in range(0, 1):
    dataset_generator = HarmonizationDatasetGenerator(
        image_cropper=ImageCropper(),
        image_paster=ImagePaster(),
        image_compositor=ImageCompositor(),
        image_shape_adjuster=TransparentImageAdjuster(),
        harmonization_mask_generator=TransparentMaskGenerator(fill=True),
        inpainting_mask_generator=TransparentMaskGenerator(fill=False, border_size=21, centered_border=True),
        transparent_image_cleaner=TransparentImageCleaner(threshold=0.4),
        inpainter=MockImageInpainter(),
        harmonizer=MockImageHarmonizer()
    )
    dataset_generator.generate(
        image=image,
        resolution=(512, 512),
        save_as="result.png"
        )