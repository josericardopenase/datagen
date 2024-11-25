from pipelines.dependencies.mmseg_api import MMSegAPI
from PIL import Image
import matplotlib.pyplot as plt
from pipelines.utils import draw_square_inside_image
from pipelines.dependencies.point_extractors.point_extractor import PointExtractor
import random
import numpy as np
from typing import List, Tuple, Callable
from PIL import Image
from scipy.stats import multivariate_normal

class MMSegPointExtractor(PointExtractor):
    def __init__(self, api: MMSegAPI):
        self.api = api

    def extract(self, image: Image.Image) -> Tuple[int, int]:
        segmented_img = self.api.segment_image(image)
        plt.imshow(segmented_img)
        plt.axis('off')
        plt.show()
        color = self.api.get_inference_color("sea")
        pixels = self.get_pixels_with_color(color, segmented_img)
        pixels_with_rules_applied = self.filter_pixels_by_rules(pixels, rules=[
        self.pixels_cannot_be_near_y_axis_edge_with_color(120, segmented_img, color),
        self.pixels_cannot_be_near_x_axis_edge_with_color(250, segmented_img, color)
        ])
        if not pixels_with_rules_applied:
            raise ValueError("No se encontraron píxeles válidos después de aplicar las reglas.")
        return self.sample_from_multivariate_normal(pixels_with_rules_applied)

    def pixels_cannot_be_near_y_axis_edge_with_color(self, margin: int, img: Image.Image, color: Tuple[int, int, int]):
        img_arr = np.array(img)

        def rule(pixel: Tuple[int, int]) -> bool:
            x, y = pixel
            if (y - margin < 0) or (y + margin >= img_arr.shape[0]):
                return False
            if not (self.is_same_color(tuple(img_arr[y - margin, x]), color) and self.is_same_color(
                    tuple(img_arr[y + margin, x]), color)):
                return False
            return True

        return rule

    def pixels_cannot_be_near_x_axis_edge_with_color(self, margin: int, img: Image.Image, color: Tuple[int, int, int]):
        img_arr = np.array(img)

        def rule(pixel: Tuple[int, int]) -> bool:
            x, y = pixel
            if (x - margin < 0) or (x + margin >= img_arr.shape[1]):
                return False
            if not (self.is_same_color(tuple(img_arr[y, x - margin]), color) and self.is_same_color(
                    tuple(img_arr[y, x + margin]), color)):
                return False
            return True

        return rule

    def get_pixels_with_color(self, color: Tuple[int, int, int], img: Image.Image) -> List[Tuple[int, int]]:
        numpy_img = np.array(img)
        height, width, _ = numpy_img.shape
        pixels = [(x, y) for y in range(height) for x in range(width) if self.is_same_color(tuple(numpy_img[y, x]), color)]
        return pixels

    @staticmethod
    def is_same_color(pixel_color, target_color):
        return pixel_color[:3] == target_color[:3]

    def filter_pixels_by_rules(self, pixels: List[Tuple[int, int]], rules: List[Callable[[Tuple[int, int]], bool]]) -> \
            List[Tuple[int, int]]:
        return [pixel for pixel in pixels if all(rule(pixel) for rule in rules)]

    def sample_from_multivariate_normal(self, pixels: List[Tuple[int, int]]) -> Tuple[int, int]:
        if not pixels:
            raise ValueError("No se pueden muestrear píxeles, ya que no hay píxeles válidos disponibles.")
        mean = np.mean(pixels, axis=0)
        cov = np.cov(np.array(pixels).T)
        sampled_point = multivariate_normal.rvs(mean=mean, cov=cov)
        sampled_pixel = tuple(map(int, map(round, sampled_point)))
        return random.choice(pixels) if sampled_pixel not in pixels else sampled_pixel

"""
api = MMSegAPI(url="http://100.103.218.9:4553/v1")
point_extractor = MMSegPointExtractor(api)
image_path = "../../../assets/bgs/bg.jpg"
img = Image.open(image_path)
pixel = point_extractor.extract(img)
highlighted_img = draw_square_inside_image(img, (500, 500), pixel, 3, 20)
plt.imshow(highlighted_img)
plt.axis('off')
plt.show()
"""