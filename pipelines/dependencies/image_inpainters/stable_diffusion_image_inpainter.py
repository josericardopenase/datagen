from diffusers import StableDiffusionInpaintPipeline
import torch
from PIL import Image
from pipelines.dependencies.image_inpainters.image_inpainter import ImageInpainter


class StableDiffusionImageInpainter(ImageInpainter):
    def __init__(
        self,
        model_id="stabilityai/stable-diffusion-2-inpainting",
        num_inference_steps=50,
        strength=0.75,
        guidance_scale=7.5,
    ):
        self.model_id = model_id
        self.num_inference_steps = num_inference_steps
        self.strength = strength
        self.guidance_scale = guidance_scale

    def inpaint(self,  original_image: Image.Image, mask_image: Image.Image, prompt : str ="") -> Image.Image:
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            self.model_id, torch_dtype=torch.float16
        )
        pipe = pipe.to("cuda")
        images = pipe(
            prompt=prompt,
            image=original_image,
            mask_image=mask_image,
        ).images
        return images[0]