from typing import Tuple
from PIL import Image
import numpy as np
from pipelines.dependencies.object_stitchers.object_stitcher import ObjectStitcher
from libcom import MureObjectStitchModel
import torch


class LibcomObjectStitcher(ObjectStitcher):
    def __init__(self, device: str = 'cpu', model_type: str = 'ObjectStitch', sampler: str = 'plms'):
        self.device = torch.device(device)
        self.model = MureObjectStitchModel(device=self.device, model_type=model_type, sampler=sampler)

    def stitch(self, bg: Image.Image, fg: Image.Image, fg_mask: Image.Image,
               bbox: Tuple[int, int, int, int]) -> Image.Image:
        # Convertir las imágenes de PIL a arrays de numpy
        bg_np = np.array(bg.convert("RGB"))
        fg_np = np.array(fg.convert("RGB"))
        fg_mask_np = np.array(fg_mask.convert("L"))

        # Convertir bbox a lista en formato [x1, y1, x2, y2]
        x1, y1, x2, y2 = bbox
        bbox_list = [x1, y1, x2, y2]

        # Ejecutar el modelo de composición con el método __call__
        result_images = self.model(
            background_image=bg_np,
            foreground_image=fg_np,
            foreground_mask=fg_mask_np,
            bbox=bbox_list,
            num_samples=1,
            sample_steps=25,
            guidance_scale=5,
            seed=321
        )

        # Convertir el resultado a una imagen PIL y devolver la primera imagen generada
        result_img = Image.fromarray(result_images[0])
        return result_img
