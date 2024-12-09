from typing import Tuple
from PIL import Image
from dataclasses import dataclass

import random
from pipelines.dependencies.background_removers.background_remover import BackgroundRemover
from pipelines.dependencies.background_removers.mmseg_background_remover import MMSegBackgroundRemover
from pipelines.dependencies.image_cropper import ImageCropper
from pipelines.dependencies.image_generators.image_generator import ImageGenerator
from pipelines.dependencies.image_generators.sthocastic_image_generator import StochasticImageGenerator
from pipelines.dependencies.image_harmonizers.image_harmonizer import ImageHarmonizer
from pipelines.dependencies.image_harmonizers.libcom_image_harmonizer import LibcomImageHarmonizer
from pipelines.dependencies.image_inpainters.image_inpainter import ImageInpainter
from pipelines.dependencies.image_inpainters.stable_diffusion_image_inpainter import StableDiffusionImageInpainter
from pipelines.dependencies.image_paster import ImagePaster
from pipelines.dependencies.loggers.logger import Logger
from pipelines.dependencies.loggers.terminal_logger import TerminalLogger
from pipelines.dependencies.api.mmseg_api import MMSegAPI
from pipelines.dependencies.point_extractors.mmseg_point_extractor import MMSegPointExtractor
from pipelines.dependencies.point_extractors.point_extractor import PointExtractor
from pipelines.dependencies.quality_evaluators.dataset_similarity_evaluators.fid_dataset_similarity_evaluator import \
    FIDDatasetSimilarityEvaluator
from pipelines.dependencies.quality_evaluators.quality_evaluator import QualityEvaluator
from pipelines.dependencies.quality_evaluators.text_image_similarity_evaluators.clip_text_image_similarity_evaluator import \
    CLIPTextImageSimilarityEvaluator
from pipelines.dependencies.quality_evaluators.image_similarity_evaluators.lpips_image_similarity_evaluator import \
    LPIPSImageSimilarityEvaluator
from pipelines.harmonization.dependencies.image_compositor import ImageCompositor
from pipelines.harmonization.dependencies.transparent_image_adjuster import TransparentImageAdjuster
from pipelines.harmonization.dependencies.transparent_image_cleaner import TransparentImageCleaner
from pipelines.harmonization.dependencies.transparent_mask_generator import TransparentMaskGenerator
from pipelines.utils import plot_images, draw_square_inside_image
import sys

@dataclass
class HarmonizationDatasetGenerator:
    point_extractor: PointExtractor
    background_image_generator: ImageGenerator
    boat_image_generator: ImageGenerator
    background_remover: BackgroundRemover
    image_cropper: ImageCropper
    image_paster: ImagePaster
    image_compositor: ImageCompositor
    image_shape_adjuster: TransparentImageAdjuster
    harmonization_mask_generator: TransparentMaskGenerator
    inpainting_inside_mask_generator: TransparentMaskGenerator
    inpainting_outside_mask_generator: TransparentMaskGenerator
    transparent_image_cleaner: TransparentImageCleaner
    inpainter: ImageInpainter
    harmonizer: ImageHarmonizer
    quality_evaluator: QualityEvaluator
    logger : Logger

    def generate(self, resolution: Tuple[int, int], save_as="result1"):
        self.logger.info("Generating background")
        background = self.background_image_generator.generate()
        self.logger.info("Generating boat")
        boat = self.boat_image_generator.generate()

        self.logger.info("Extracting point of possible boat")
        boat_position = self.point_extractor.extract(background)
        self.logger.info("Removing background of boat")
        boat_without_background = self.background_remover.remove(boat)

        self.logger.info("Making image composition of background and boat")
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
            size_of=0.55
        )

        self.logger.info("Harmonizing boat")

        harmonization_mask = self.generate_harmonization_mask(cleaned_boat, background_cropped_image)
        harmonized_image = self.harmonizer.harmonize(composited_image, harmonization_mask)
        self.logger.info("Inpainting boat borders")
        inpainting_mask, fg_shape = self.generate_inpainting_mask(cleaned_boat, background_cropped_image, fg_shape)
        prompt = "A boat"
        inpainted_image = self.inpainter.inpaint(harmonized_image, inpainting_mask, prompt=prompt)
        inpainted_image = self.inpainter.inpaint(inpainted_image, inpainting_mask, prompt=prompt)
        pasted = self.image_paster.paste(
            original_image=background,
            pasted_image=inpainted_image,
            center=boat_position
        )
        self.logger.info("Generating insights of process")
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
        #self.quality_evaluator.evaluate_text_image_similarity(prompt, inpainted_image)
        #self.quality_evaluator.evaluate_image_similarity(background, pasted)
        #self.quality_evaluator.show_scores()
        return pasted

    def generate_inpainting_mask(self, cleaned_boat, cropped_image, fg_shape):
        composited_inpainting_mask, fg_shape = self.image_compositor.composite(
            background=Image.new("RGB", cropped_image.size, color=(0, 0, 0)),
            foreground= self.inpainting_inside_mask_generator.generate(cleaned_boat),
            center=(cropped_image.size[0] // 2, cropped_image.size[1] // 2),
            size_of=0.55
        )
        return composited_inpainting_mask, fg_shape

    def generate_outside_inpainting_mask(self, cleaned_boat, cropped_image, fg_shape):
        composited_inpainting_mask, fg_shape = self.image_compositor.composite(
            background=Image.new("RGB", cropped_image.size, color=(0, 0, 0)),
            foreground= self.inpainting_inside_mask_generator.generate(cleaned_boat),
            center=(cropped_image.size[0] // 2, cropped_image.size[1] // 2),
            size_of=0.55
        )
        return composited_inpainting_mask, fg_shape

    def generate_harmonization_mask(self, cleaned_boat, cropped_image):
        composited_harmonization_mask, fg_shape = self.image_compositor.composite(
            background=Image.new("RGB", cropped_image.size, color=(0, 0, 0)),
            foreground=self.harmonization_mask_generator.generate(cleaned_boat),
            center=(cropped_image.size[0] // 2, cropped_image.size[1] // 2),
            size_of=0.55
        )
        return composited_harmonization_mask


folder = sys.argv[0] if sys.argv[0] else 0

for x in range(0, 500):
    try:
        dataset_generator = HarmonizationDatasetGenerator(
            point_extractor=MMSegPointExtractor(MMSegAPI(url="http://100.103.218.9:4553/v1")),
            background_image_generator=StochasticImageGenerator("assets/bgs"),
            boat_image_generator=StochasticImageGenerator("assets/boats/with_bg"),
            background_remover=MMSegBackgroundRemover("ship",
                                                      MMSegAPI(url="http://100.103.218.9:4553/v1")
                                                      ),
            image_cropper=ImageCropper(),
            image_compositor=ImageCompositor(),
            image_shape_adjuster=TransparentImageAdjuster(),
            transparent_image_cleaner=TransparentImageCleaner(threshold=0.4),
            harmonization_mask_generator=TransparentMaskGenerator(fill=True),
            harmonizer=LibcomImageHarmonizer(),
            inpainting_inside_mask_generator=TransparentMaskGenerator(fill=False, border_size=31, inside_border=True),
            inpainting_outside_mask_generator=TransparentMaskGenerator(fill=False, border_size=97),
            inpainter=StableDiffusionImageInpainter(),
            image_paster=ImagePaster(),
            quality_evaluator=QualityEvaluator(
                image_similarity=LPIPSImageSimilarityEvaluator(),
                text_image_similarity= CLIPTextImageSimilarityEvaluator(),
                aesthetic_eval=None,
                dataset_similarity=FIDDatasetSimilarityEvaluator()
            ),
            logger=TerminalLogger()
        )
        result = dataset_generator.generate((512, 512), f's_dataset/result_{x}_process.png')
        result.save(f's_dataset/result_{x}.png')
    except Exception as inst:
        print(inst)
        
