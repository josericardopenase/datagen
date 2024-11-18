import requests
import base64

# Function to convert an image to Base64
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Replace with the path to your image file
image_path = "assets/bgs/bg.jpg"
image_base64 = encode_image_to_base64(image_path)

# Define the URL of your FastAPI endpoint
url = "http://100.103.218.9:4553/v1/inference/images"  # Update this with your actual endpoint URL

# Request payload
payload = {
    "config_engine": "../configs/mask2former/mask2former_swin-b-in1k-384x384-pre_8xb2-160k_ade20k-640x640.py",
    "framework": "pytorch",
    "checkpoint": "../val_da_best_mIoU_iter_16000.pth",  # Set to a string value if required
    "output_prefix": "result-",
    "device": "cuda:2",
    "opacity": 0.5,
    "title": "result",
    "labels": "True",
    "custom_ops": None,  # Set to a string value if required
    "input_size": None,  # Set to an integer value if required
    "precision": None,   # Set to a string value if required
    "images_base64": [image_base64, image_base64, image_base64]
}

# Send the POST request
response = requests.post(url, json=payload)

# Check the response
if response.status_code == 200:
    print("Success:", response.json())
else:
    print("Error:", response.status_code, response.text)



