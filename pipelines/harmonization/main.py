from typing import Tuple
from PIL import Image

from pipelines.dependencies.background_removers.background_remover import BackgroundRemover
from pipelines.dependencies.background_removers.mock_background_remover import MockBackgroundRemover
from pipelines.dependencies.image_cropper import ImageCropper
from pipelines.dependencies.image_generators.MockImageGenerator import MockImageGenerator
from pipelines.dependencies.image_generators.image_generator import ImageGenerator
from pipelines.dependencies.image_harmonizers.image_harmonizer import ImageHarmonizer
from pipelines.dependencies.image_harmonizers.libcom_image_harmonizer import LibcomImageHarmonizer
from pipelines.dependencies.image_inpainters.image_inpainter import ImageInpainter
from pipelines.dependencies.image_inpainters.stable_diffusion_image_inpainter import StableDiffusionImageInpainter
from pipelines.dependencies.image_paster import ImagePaster
from pipelines.dependencies.point_extractors.mock_point_extractor import MockPointExtractor
from pipelines.dependencies.point_extractors.point_extractor import PointExtractor
from pipelines.harmonization.dependencies.image_compositor import ImageCompositor
from pipelines.harmonization.dependencies.transparent_image_adjuster import TransparentImageAdjuster
from pipelines.harmonization.dependencies.transparent_image_cleaner import TransparentImageCleaner
from pipelines.harmonization.dependencies.transparent_mask_generator import TransparentMaskGenerator
from pipelines.utils import plot_images, draw_square_inside_image
import sys

class HarmonizationDatasetGenerator:
    def __init__(self,
                 point_extractor : PointExtractor,
                 background_image_generator: ImageGenerator,
                 boat_image_generator : ImageGenerator,
                 background_remover : BackgroundRemover,
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
        self.background_image_generator = background_image_generator
        self.boat_image_generator = boat_image_generator
        self.image_cropper = image_cropper
        self.image_paster = image_paster
        self.image_compositor = image_compositor
        self.image_shape_adjuster = image_shape_adjuster
        self.harmonization_mask_generator = harmonization_mask_generator
        self.inpainting_mask_generator = inpainting_mask_generator
        self.transparent_image_cleaner = transparent_image_cleaner
        self.inpainter = inpainter
        self.harmonizer = harmonizer
        self.point_extractor = point_extractor
        self.background_remover = background_remover


    def generate(self, resolution: Tuple[int, int], save_as="result1"):
        background = self.background_image_generator.generate()
        boat = self.boat_image_generator.generate()

        boat_position = self.point_extractor.extract(background)
        boat_without_background = self.background_remover.remove(boat)

        background_cropped_image = self.image_cropper.crop(
            image=background,
            center=boat_position,
            resolution=resolution
        )
        adjusted_boat = self.image_shape_adjuster.adjust(boat_without_background, 1)
        cleaned_boat = self.transparent_image_cleaner.clean(adjusted_boat)
        composited_image, fg_shape = self.image_compositor.composite(
            background=background_cropped_image,
            foreground=cleaned_boat,
            center=(background_cropped_image.size[0]//2, background_cropped_image.size[1]//2),
            size_of=0.4
        )

        harmonization_mask = self.generate_harmonization_mask(cleaned_boat, background_cropped_image)
        harmonized_image = self.harmonizer.harmonize(composited_image, harmonization_mask)
        inpainting_mask, fg_shape = self.generate_inpainting_mask(cleaned_boat, background_cropped_image, fg_shape)
        inpainted_image = self.inpainter.inpaint(harmonized_image, inpainting_mask, prompt="A boat")
        pasted = self.image_paster.paste(
            original_image=background,
            pasted_image=inpainted_image,
            center=boat_position
        )
        plot_images(
            [
                background,
                boat,
                draw_square_inside_image(background, background_cropped_image.size, boat_position, border_width=7, center_radius=10),
                background_cropped_image,
                composited_image,
                harmonization_mask,
                harmonized_image,
                inpainting_mask,
                inpainted_image,
                pasted,
                draw_square_inside_image(pasted, fg_shape, boat_position, border_width=7, center_radius=10)
            ],
            ["Imagen original", "Barco original", "Posición de recorte", "Recorte", "Barco incluído", "Máscara de harmonización","Imagen harmonizada",  "Máscara de inpainting", "Imagen con inpainting realizado", "Imagen original con región copiada", "Bounding box añadida"],
            main_title="Pipeline using Image Harmonization",
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

for iteration in range(0, 1):
    dataset_generator = HarmonizationDatasetGenerator(
        point_extractor=MockPointExtractor((450, 450)),
        background_image_generator=MockImageGenerator("assets/bgs/bg.jpg"),
        boat_image_generator=MockImageGenerator("assets/boats/boat.png"),
        background_remover=MockBackgroundRemover(),
        image_cropper=ImageCropper(),
        image_compositor=ImageCompositor(),
        image_shape_adjuster=TransparentImageAdjuster(),
        transparent_image_cleaner=TransparentImageCleaner(threshold=0.4),
        harmonization_mask_generator=TransparentMaskGenerator(fill=True),
        harmonizer=LibcomImageHarmonizer(),
        inpainting_mask_generator=TransparentMaskGenerator(fill=False, border_size=21, inside_border=True),
        inpainter=StableDiffusionImageInpainter(),
        image_paster=ImagePaster()
    )

    dataset_generator.generate(
        resolution=(512, 512),
        save_as="result.png"
        )