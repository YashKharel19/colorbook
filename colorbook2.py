import os
from PIL import Image, ImageFilter, ImageOps
import matplotlib.pyplot as plt

# Input and output folder paths
input_folder = r"G:\Lumasha all\colorbook code\HindiColorbook"
output_folder = r"G:\Lumasha all\colorbook code\Outlined"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Supported image formats
supported_formats = ('.png', '.jpg', '.jpeg')

# Process each image in the input folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith(supported_formats):
        image_path = os.path.join(input_folder, filename)
        image = Image.open(image_path)

        # Convert image to grayscale
        gray_image = image.convert("L")

        # Apply edge detection and invert
        edges = gray_image.filter(ImageFilter.FIND_EDGES)
        inverted_edges = ImageOps.invert(edges)

        # Save the outlined image
        output_path = os.path.join(output_folder, f"outlined_{filename}")
        inverted_edges.save(output_path)

print("All images have been processed and saved to the Outlined folder.")
