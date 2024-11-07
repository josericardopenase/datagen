from PIL import Image
from matplotlib import pyplot as plt
from pipelines.inpainting.dependencies.image_cropper import ImageCropper
from pipelines.inpainting.dependencies.image_paster import ImagePaster
from pipelines.inpainting.dependencies.mask_creator import MaskCreator

image = Image.open("assets/bgs/bg.jpg")
shape = Image.open("assets/masks/square_mask.png")

cropper = ImageCropper(image)
cropped_image = cropper.crop((450, 450), (400, 400))
plt.imshow(cropped_image)
plt.show()

mask_creator = MaskCreator(shape=shape)
mask = mask_creator.create((150, 150), (300, 300), size_of_shape=0.05)

plt.imshow(mask)
plt.show()

image_paster = ImagePaster(original_image=image, pasted_image=mask)
pasted = image_paster.paste((450, 450))
plt.imshow(pasted)
plt.show()



