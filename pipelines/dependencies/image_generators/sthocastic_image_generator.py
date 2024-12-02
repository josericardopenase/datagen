import random
from PIL import Image
import os

from pipelines.dependencies.image_generators.image_generator import ImageGenerator


class StochasticImageGenerator(ImageGenerator):
    def __init__(self, dir: str):
        self.dir = dir


    def is_image(self, path : str) -> bool:
        return path.endswith('.jpg') or path.endswith('.png')

    def generate(self) -> Image.Image:
        images = list(
            map(lambda img : Image.open(f'{self.dir}/{img}') , filter(lambda f: self.is_image(f), os.listdir(self.dir)))
        )
        number = random.randint(0, len(images) - 1)
        return images[number]
