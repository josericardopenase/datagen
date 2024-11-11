from PIL import Image
from abc import ABC, abstractmethod

from pipelines.dependencies.image_inpainters.image_inpainter import ImageInpainter


class MockImageInpainter(ImageInpainter):
    def inpaint(self,  original_image: Image.Image, mask_image: Image.Image, prompt : str ="") -> Image.Image:
        return original_image
