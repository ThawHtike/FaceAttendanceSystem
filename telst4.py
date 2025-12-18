import requests
from PIL import Image
from io import BytesIO

# 1. URL of the official test image (Joe Biden)
url = "https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/biden.jpg"

print("Downloading sample image...")

try:
    # 2. Download the image data
    response = requests.get(url)

    if response.status_code == 200:
        # 3. Open it with PIL
        img = Image.open(BytesIO(response.content))

        # 4. Force it to be standard RGB (Standard JPG format)
        img = img.convert("RGB")

        # 5. Save it to your folder
        save_name = "sample_face.jpg"
        img.save(save_name, "JPEG", quality=100)

        print(f"✅ Success! Saved '{save_name}' in your project folder.")
        print("You can now upload this file in your Streamlit app.")

    else:
        print("Error: Could not reach the internet to download the image.")

except Exception as e:
    print(f"Error: {e}")