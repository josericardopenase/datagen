from PIL import Image
from matplotlib import pyplot as plt
from pipelines.harmonization.dependencies.transparent_mask_generator import TransparentMaskGenerator


def test_transparent_mask_generator():
    mask_generator = TransparentMaskGenerator(fill=False, inside_border=False, border_size=40)
    fg = Image.open("assets/boat.png")
    plt.imshow(mask_generator.generate(fg, (512, 512)))
    plt.show()