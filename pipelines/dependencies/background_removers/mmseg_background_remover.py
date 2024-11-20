from PIL import Image

import numpy as np
from matplotlib import pyplot as plt

from pipelines.dependencies.background_removers.background_remover import BackgroundRemover
from pipelines.dependencies.mmseg_api import MMSegAPI


class MMSegBackgroundRemover(BackgroundRemover):
    def __init__(self, category: str, api : MMSegAPI):
        self.category = category
        self.api = api

    def remove(self, image: Image.Image) -> Image.Image:
        segmented_image = self.api.segment_image(image)
        color = self.api.get_inference_color(self.category)
        arr_segmented_image = np.array(segmented_image)
        final_image = np.array(image.convert("RGBA"))
        for x in range(0, arr_segmented_image.shape[0]):
            for y in range(0, arr_segmented_image.shape[1]):
                if not self.is_same_color(arr_segmented_image[x, y], color):
                    final_image[x, y] = [0, 0, 0, 0]
        
        
        return self.clean_small_islands(final_image)

    @staticmethod
    def is_same_color(pixel_color, target_color):
        return np.array_equal(pixel_color[:3], target_color[:3])

    def clean_small_islands(self, final_image, min_size=500):
        visited = set()
        target_color = [0, 0, 0]  # Adjust based on the background color used

        for x in range(final_image.shape[0]):
            for y in range(final_image.shape[1]):
                # Skip if pixel is already visited or transparent
                if (x, y) in visited or np.array_equal(final_image[x, y], [0, 0, 0, 0]):
                    continue

                # Calculate area of the connected component
                area = self.area_of_pixel_is_less_than(final_image, x, y, visited, target_color)

                # Delete small islands
                if area < min_size:
                    self.delete_area(final_image, x, y, visited, target_color)

        return Image.fromarray(final_image, "RGBA")

    def area_of_pixel_is_less_than(self, final_image, x, y, visited, target_color) -> int:
        # Boundary check
        if x < 0 or y < 0 or x >= final_image.shape[0] or y >= final_image.shape[1]:
            return 0

        # If pixel is already visited or not part of the target region
        if (x, y) in visited or not np.array_equal(final_image[x, y][:3], target_color):
            return 0

        # Mark the pixel as visited
        visited.add((x, y))

        # Recursive flood-fill to calculate area
        area = 1
        area += self.area_of_pixel_is_less_than(final_image, x + 1, y, visited, target_color)
        area += self.area_of_pixel_is_less_than(final_image, x - 1, y, visited, target_color)
        area += self.area_of_pixel_is_less_than(final_image, x, y + 1, visited, target_color)
        area += self.area_of_pixel_is_less_than(final_image, x, y - 1, visited, target_color)

        return area

    def delete_area(self, final_image, x, y, visited, target_color):
        # Boundary check
        if x < 0 or y < 0 or x >= final_image.shape[0] or y >= final_image.shape[1]:
            return

        # If pixel is already visited or not part of the target region
        if (x, y) in visited or not np.array_equal(final_image[x, y][:3], target_color):
            return

        # Mark the pixel as visited
        visited.add((x, y))

        # Set pixel to transparent
        final_image[x, y] = [0, 0, 0, 0]

        # Recursive flood-fill to delete area
        self.delete_area(final_image, x + 1, y, visited, target_color)
        self.delete_area(final_image, x - 1, y, visited, target_color)
        self.delete_area(final_image, x, y + 1, visited, target_color)
        self.delete_area(final_image, x, y - 1, visited, target_color)

