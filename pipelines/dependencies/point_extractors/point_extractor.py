from PIL import Image
from abc import ABC, abstractmethod


class PointExtractor(ABC):
    @abstractmethod
    def extract(self, image : Image.Image):
        ...