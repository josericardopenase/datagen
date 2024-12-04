from PIL import Image

from matplotlib import pyplot as plt
from pipelines.dependencies.api.mmseg_api import MMSegAPI


def test_mmseg_api_image_is_segmented():
    img = Image.open("assets/boats/boat3.png")
    segmentation = MMSegAPI(url="http://100.103.218.9:4553/v1").segment_image(img)
    plt.imshow(segmentation)
    plt.show()
    assert segmentation is not None

def test_mmseg_api_image_is_cached():
    img = Image.open("assets/boats/boat3.png")
    segmentation = MMSegAPI(url="http://100.103.218.9:4553/v1").segment_image(img)
    plt.imshow(segmentation)
    plt.show()
    assert segmentation is not None
