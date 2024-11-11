from PIL import Image
from abc import ABC, abstractmethod


class ImageInpainter(ABC):
    @abstractmethod
    def inpaint(self,  original_image: Image.Image, mask_image: Image.Image, prompt : str ="") -> Image.Image:
        ...
