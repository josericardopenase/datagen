from abc import abstractmethod, ABC

from PIL  import Image

from pipelines.dependencies.image_harmonizers.image_harmonizer import ImageHarmonizer


class MockImageHarmonizer(ImageHarmonizer):
    def harmonize(self, image : Image.Image, mask: Image.Image) -> Image.Image:
        return image
