from abc import abstractmethod
from PIL import Image


class ImageGenerator:
    @abstractmethod
    def generate(self) -> Image.Image:
        ...