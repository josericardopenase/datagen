from abc import abstractmethod, ABC
from PIL import Image


class BackgroundRemover(ABC):
    @abstractmethod
    def remove(self, image : Image.Image) -> Image.Image:
        ...