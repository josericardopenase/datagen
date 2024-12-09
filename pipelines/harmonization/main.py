import sys
from pipelines.dependencies.background_removers.mmseg_background_remover import MMSegBackgroundRemover
from pipelines.dependencies.dataset_savers.yolo_dataset_saver import YoloDatasetSaver
from pipelines.dependencies.image_generators.sthocastic_image_generator import StochasticImageGenerator
from pipelines.dependencies.image_harmonizers.libcom_image_harmonizer import LibcomImageHarmonizer
from pipelines.dependencies.loggers.terminal_logger import TerminalLogger
from pipelines.dependencies.api.mmseg_api import MMSegAPI
from pipelines.dependencies.point_extractors.mmseg_point_extractor import MMSegPointExtractor
from pipelines.dependencies.image_inpainters.stable_diffusion_image_inpainter import StableDiffusionImageInpainter
from pipelines.dependencies.quality_evaluators.dataset_similarity_evaluators.fid_dataset_similarity_evaluator import \
    FIDDatasetSimilarityEvaluator
from pipelines.dependencies.quality_evaluators.text_image_similarity_evaluators.clip_text_image_similarity_evaluator import \
    CLIPTextImageSimilarityEvaluator
from pipelines.dependencies.quality_evaluators.image_similarity_evaluators.lpips_image_similarity_evaluator import \
    LPIPSImageSimilarityEvaluator
from pipelines.dependencies.image_cropper import ImageCropper
from pipelines.dependencies.image_paster import ImagePaster
from pipelines.dependencies.quality_evaluators.quality_evaluator import QualityEvaluator
from pipelines.harmonization.dependencies.image_compositor import ImageCompositor
from pipelines.harmonization.dependencies.transparent_image_adjuster import TransparentImageAdjuster
from pipelines.harmonization.dependencies.transparent_image_cleaner import TransparentImageCleaner
from pipelines.harmonization.dependencies.transparent_mask_generator import TransparentMaskGenerator
from pipelines.harmonization.harmonization_dataset_generator import HarmonizationDatasetGenerator

folder = sys.argv[0] if sys.argv[0] else 0
dataset_generator = HarmonizationDatasetGenerator(
    point_extractor=MMSegPointExtractor(MMSegAPI(url="http://100.103.218.9:4553/v1")),
    background_image_generator=StochasticImageGenerator("assets/"),
    boat_image_generator=StochasticImageGenerator("assets/boats/"),
    background_remover=MMSegBackgroundRemover("ship",
                                              MMSegAPI(url="http://100.103.218.9:4553/v1")
                                              ),
    harmonization_mask_generator=TransparentMaskGenerator(fill=True),
    harmonizer=LibcomImageHarmonizer(),
    inpainting_inside_mask_generator=TransparentMaskGenerator(fill=False, border_size=31, inside_border=True),
    inpainting_outside_mask_generator=TransparentMaskGenerator(fill=False, border_size=97),
    inpainter=StableDiffusionImageInpainter(),
    transparent_image_cleaner=TransparentImageCleaner(threshold=0.4),
    image_paster=ImagePaster(),
    image_cropper=ImageCropper(),
    image_compositor=ImageCompositor(),
    image_shape_adjuster=TransparentImageAdjuster(),
    quality_evaluator=QualityEvaluator(
        image_similarity=LPIPSImageSimilarityEvaluator(),
        text_image_similarity=CLIPTextImageSimilarityEvaluator(),
        aesthetic_eval=None,
        dataset_similarity=FIDDatasetSimilarityEvaluator()
    ),
    logger=TerminalLogger()
)
dataset_saver = YoloDatasetSaver(boat_category=0)

for x in range(0, 1):
    generated_image, bounding_box = dataset_generator.generate((512, 512), f's_dataset/result_{x}_process.png')
    generated_image.save(f's_dataset/result_{x}.png')
    dataset_saver.add_training(generated_image, bounding_box)

dataset_saver.save(".")

