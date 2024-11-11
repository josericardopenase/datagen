from abc import abstractmethod, ABC

from PIL  import Image

class ImageHarmonizer(ABC):
    @abstractmethod
    def harmonize(self, image : Image.Image, mask: Image.Image) -> Image.Image:
        ...
