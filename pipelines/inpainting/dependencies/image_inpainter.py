from diffusers import StableDiffusionInpaintPipeline
import torch
from PIL import Image

class StableDiffusionImageInpainter:
    def __init__(
        self,
        prompt,
        original_image: Image.Image,
        mask_image: Image.Image,
        model_id="stabilityai/stable-diffusion-2-inpainting",
        num_inference_steps=50,
        strength=0.75,
        guidance_scale=7.5,
    ):
        self.original_image = original_image
        self.mask_image = mask_image
        self.model_id = model_id
        self.num_inference_steps = num_inference_steps
        self.strength = strength
        self.guidance_scale = guidance_scale
        self.prompt = prompt

    def inpaint(self, prompt=""):
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            self.model_id, torch_dtype=torch.float16
        )
        pipe = pipe.to("cuda")  # Move the model to GPU
        images = pipe(
            prompt=self.prompt,
            image=self.original_image,  # The original image
            mask_image=self.mask_image,  # The mask that indicates areas to be inpainted
        ).images
        return images