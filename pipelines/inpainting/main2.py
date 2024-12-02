from dataclasses import dataclass
from pipelines.dependencies.image_generators.MockImageGenerator import MockImageGenerator
from pipelines.dependencies.image_generators.image_generator import ImageGenerator
from pipelines.dependencies.image_inpainters.stable_diffusion_image_inpainter import StableDiffusionImageInpainter, ImageInpainter
from pipelines.inpainting.dependencies.mask_creator import MaskCreator
from pipelines.utils import plot_images, draw_square_inside_image
import sys

@dataclass
class InpaintingDatasetGenerator:
    image_generator: ImageGenerator
    inpainter: ImageInpainter

    def generate(self, save_as="result1"):
        bg = self.image_generator.generate()
        resolution = (bg.size[0] - 100, bg.size[1] - 100)
        point = (bg.size[0] // 2, bg.size[1] // 2)
        mask = MaskCreator(
            resolution_of_shape=resolution
        ).create(
            center=point,
            resolution=(bg.size[0], bg.size[1])
        )
        inpainted_image = self.inpainter.inpaint(
            prompt="a boat crossing the sea",
            original_image=bg,
            mask_image=mask
        )
        plot_images(
            [
                bg,
                draw_square_inside_image(bg, resolution, point, border_width=7, center_radius=10),
                mask,
                inpainted_image,
            ],
            ["Imágen original", "Posición de recorte", "Recorte", "Mascara generada", "Mascara aplicada", "Inpainted Image", "Finally pasted image", "Máscara de segmentación"],
            main_title="Pipeline",
            save_as=save_as
        )
        return inpainted_image

folder = sys.argv[0] if sys.argv[0] else 0
for iteration in range(0, 1):
    dataset_generator = InpaintingDatasetGenerator(
        inpainter=StableDiffusionImageInpainter(),
        image_generator=MockImageGenerator('assets/bgs/bg3.jpg')

    )
    dataset_generator.generate(save_as='result_{}.png'.format( iteration))
