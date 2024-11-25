from typing import List, Any, Tuple
from io import BytesIO
from PIL import Image
import base64
import requests

class MMSegAPI:
    def __init__(self, url: str):
        self.url = url

    @staticmethod
    def to_base64(image: Image.Image) -> str:
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def segment_image(self, image: Image.Image) -> Image.Image:
        payload = self.create_payload([self.to_base64(image)])
        response = requests.post(self.url + "/inference/images", json=payload)
        b64_img = response.json()["segmented_images"][0]
        return Image.open(BytesIO(base64.b64decode(b64_img)))

    @staticmethod
    def create_payload(images: List[str]) -> dict:
        return {
            "config_engine": "/mmsegmentation/work_dirs/mask2former_swin-b-in22k-384x384-pre_8xb2-160k_ade20k-640x640/mask2former_swin-b-in22k-384x384-pre_8xb2-160k_ade20k-640x640.py",
            "framework": "pytorch",
            "checkpoint": "/mmsegmentation/work_dirs/mask2former_swin-b-in22k-384x384-pre_8xb2-160k_ade20k-640x640/val_da_best_mIoU_iter_16000.pth",
            "output_prefix": "result-",
            "device": "cuda:0",
            "opacity": 1,
            "title": "result",
            "labels": "False",
            "custom_ops": None,
            "input_size": None,
            "precision": None,
            "images_base64": images
        }

    def get_inference_color(self, category: str) -> Tuple[int, int, int]:
        response = requests.get(self.url + "/inference_colors/")
        response.raise_for_status()
        return tuple(response.json()[category])