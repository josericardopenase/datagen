from functools import reduce
import random
from matplotlib import pyplot as plt
from urllib3 import request
import numpy as np
from pipelines.dependencies.point_extractors.point_extractor import PointExtractor
from typing import List, Tuple, Callable
from io import BytesIO
from PIL import Image
import requests
import base64
from scipy.stats import multivariate_normal

from pipelines.utils import draw_square_inside_image


class MMSegPointExtractor(PointExtractor):
    def __init__(self,
                 url: str = "http://100.103.218.9:4553/v1"):
        self.url = url

    @staticmethod
    def to_base64(image: Image.Image) -> str:
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def extract(self, image: Image.Image) -> Tuple[int, int]:
        img = self.segment_image(image)
        color = self.get_inference_color("sea")
        pixels = self.get_pixels_with_color(color, img)
        pixels_with_rules_applied = self.filter_pixels_by_rules(pixels, rules=[
            self.pixels_cannot_be_near_y_axis_edge_with_color(120, img, color),
            self.pixels_cannot_be_near_x_axis_edge_with_color(250, img, color)
        ])
        if not pixels_with_rules_applied:
            raise ValueError("No se encontraron píxeles válidos después de aplicar las reglas.")
        sampled_pixel = self.sample_from_multivariate_normal(pixels_with_rules_applied)
        return sampled_pixel

    def segment_image(self, image: Image.Image):
        payload = self.create_payload([self.to_base64(image), ])
        response = requests.post(self.url + "/inference/images", json=payload)
        b64_img = response.json()["segmented_images"][0]
        img = Image.open(BytesIO(base64.b64decode(b64_img)))
        return img

    @staticmethod
    def create_payload(images: List[str]):
        payload = {
            "config_engine": "/mmsegmentation/work_dirs/mask2former_swin-b-in22k-384x384-pre_8xb2-160k_ade20k-640x640/mask2former_swin-b-in22k-384x384-pre_8xb2-160k_ade20k-640x640.py",
            "framework": "pytorch",
            "checkpoint": "/mmsegmentation/work_dirs/mask2former_swin-b-in22k-384x384-pre_8xb2-160k_ade20k-640x640/val_da_best_mIoU_iter_16000.pth",
            # Set to a string value if required
            "output_prefix": "result-",
            "device": "cuda:0",
            "opacity": 1,
            "title": "result",
            "labels": "False",
            "custom_ops": None,  # Set to a string value if required
            "input_size": None,  # Set to an integer value if required
            "precision": None,  # Set to a string value if required
            "images_base64": images
        }
        return payload

    def get_inference_color(self, category: str):
        response = requests.get(self.url + "/inference_colors/")
        if response.status_code != 200: return []
        color = response.json()[category]
        return color

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
        pixels = []
        for y in range(height):
            for x in range(width):
                if self.is_same_color(tuple(numpy_img[y, x]), color):
                    pixels.append((x, y))
        return pixels

    def is_same_color(self, param, color):
        return param[0] == color[0] and param[1] == color[1] and param[2] == color[2]

    def filter_pixels_by_rules(self, pixels: List[Tuple[int, int]], rules: List[Callable[[Tuple[int, int]], bool]]) -> \
            List[Tuple[int, int]]:
        final_pixels = []
        for pixel in pixels:
            add = True
            for rule in rules:
                if not rule(pixel):
                    add = False
            if add: final_pixels.append(pixel)
        return final_pixels

    def sample_from_multivariate_normal(self, pixels: List[Tuple[int, int]]) -> Tuple[int, int]:
        if not pixels:
            raise ValueError("No se pueden muestrear píxeles, ya que no hay píxeles válidos disponibles.")

        mean_x = np.mean([pixel[0] for pixel in pixels])
        mean_y = np.mean([pixel[1] for pixel in pixels])
        mean = [mean_x, mean_y]

        # Calculate the covariance matrix
        cov = np.cov(np.array(pixels).T)

        # Sample a point from the multivariate normal distribution
        sampled_point = multivariate_normal.rvs(mean=mean, cov=cov)

        # Round to the nearest integer pixel coordinates
        sampled_pixel = (int(round(sampled_point[0])), int(round(sampled_point[1])))

        # Ensure the sampled pixel is within the bounds of the filtered pixels
        sampled_pixel = (
            min(max(sampled_pixel[0], min([pixel[0] for pixel in pixels])), max([pixel[0] for pixel in pixels])),
            min(max(sampled_pixel[1], min([pixel[1] for pixel in pixels])), max([pixel[1] for pixel in pixels]))
        )

        # Check if the sampled pixel still meets the rules
        if sampled_pixel not in pixels:
            sampled_pixel = random.choice(pixels)  # Fall back to choosing a valid pixel randomly if sampling failed

        return sampled_pixel


"""
img = Image.open("../../../assets/bgs/bg.jpg")
point_extractor = MMSegPointExtractor()
pixel = point_extractor.extract(img)
print(pixel)

plt.imshow(draw_square_inside_image(img, (500, 500), pixel, 3, 20))
plt.show()
"""