from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter


class SegmentationMaskGenerator:
    def __init__(self, threshold: float, block_size: int = 8):
        self.threshold = threshold
        self.block_size = block_size

    def generate(self, original_image: Image.Image, inpainted_image: Image.Image) -> Image.Image:
        original_array = np.array(original_image.convert("RGBA"), dtype=np.float32)
        inpainted_array = np.array(inpainted_image.convert("RGBA"), dtype=np.float32)

        # Aplicar filtro gaussiano para suavizar
        smoothed_original = gaussian_filter(original_array, sigma=(self.block_size, self.block_size, 0))
        smoothed_inpainted = gaussian_filter(inpainted_array, sigma=(self.block_size, self.block_size, 0))

        # Calcular la diferencia en cada canal RGBA
        diff = np.abs(smoothed_original - smoothed_inpainted)
        avg_diff = np.mean(diff, axis=2)  # Promedio de la diferencia en los canales

        # Crear la máscara basada en el umbral
        result_mask = (avg_diff > self.threshold).astype(np.uint8) * 255
        result_image = Image.fromarray(result_mask, mode="L")

        return result_image
