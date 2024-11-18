from pipelines.dependencies.image_generators.image_generator import ImageGenerator
from PIL import Image


class MockImageGenerator(ImageGenerator):
    def __init__(self, route : str):
        self.route = route
    def generate(self) -> Image.Image:
        return Image.open(self.route)