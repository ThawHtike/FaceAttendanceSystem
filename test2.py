from PIL import Image
import os

folder = "IMAGE_FILES"

for file in os.listdir(folder):
    path = os.path.join(folder, file)
    try:
        img = Image.open(path)
        img = img.convert("RGB")
        img.save(path, "JPEG", quality=95)
        print(f"Fixed: {file}")
    except Exception as e:
        print(f"Cannot fix {file}: {e}")
