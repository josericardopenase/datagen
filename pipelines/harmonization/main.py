import sys
from pipelines.dependencies.background_removers.mmseg_background_remover import MMSegBackgroundRemover
from pipelines.dependencies.background_removers.mock_background_remover import MockBackgroundRemover
from pipelines.dependencies.dataset_savers.yolo_dataset_saver import YoloDatasetSaver
from pipelines.dependencies.image_generators.sthocastic_image_generator import StochasticImageGenerator
from pipelines.dependencies.image_harmonizers.libcom_image_harmonizer import LibcomImageHarmonizer
from pipelines.dependencies.loggers.terminal_logger import TerminalLogger
from pipelines.dependencies.api.mmseg_api import MMSegAPI
from pipelines.dependencies.point_extractors.mmseg_point_extractor import MMSegPointExtractor
from pipelines.dependencies.image_inpainters.stable_diffusion_image_inpainter import StableDiffusionImageInpainter
from pipelines.dependencies.image_inpainters.dalle2_image_inpainter import Dalle2ImageInpainter
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
from timeit import default_timer as timer

dataset_generator = HarmonizationDatasetGenerator(
    point_extractor=MMSegPointExtractor(MMSegAPI(url="http://100.103.218.9:4553/v1")),
    background_image_generator=StochasticImageGenerator("assets/bgs"),
    boat_image_generator=StochasticImageGenerator("assets/boats/without_bg"),
    background_remover=MockBackgroundRemover(),
    harmonization_mask_generator=TransparentMaskGenerator(fill=True),
    harmonizer=LibcomImageHarmonizer(),
    inpainting_inside_mask_generator=TransparentMaskGenerator(fill=False, border_size=17, inside_border=True),
    inpainting_outside_mask_generator=TransparentMaskGenerator(fill=False, border_size=17),
    inpainter=Dalle2ImageInpainter(),
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

start = timer()
for x in range(0, 1):
    try:
        generated_image, bounding_box = dataset_generator.generate((512, 512), f's_dataset/result_{x}_process.png')
        generated_image.save(f'result_{x}.png')
        normalized_bounding_box = (bounding_box[0]/generated_image.size[0],
                                   bounding_box[1]/generated_image.size[1],
                                   bounding_box[2]/generated_image.size[0],
                                   bounding_box[3]/generated_image.size[1])
        dataset_saver.add_training(generated_image.convert("RGB"), normalized_bounding_box)
    except:
        ...

end = timer()
print("Time of creation: ", end-start)
dataset_saver.save("yolo_dataset")
