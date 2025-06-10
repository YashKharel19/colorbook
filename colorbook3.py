import os
from PIL import Image, ImageFilter, ImageOps

# Input and output folder paths
input_folder = r"G:\Lumasha all\colorbook\Nepali"
output_folder = r"G:\Lumasha all\colorbook\NepaliOutlined"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Supported image formats
supported_formats = ('.png', '.jpg', '.jpeg')

# Threshold value to clean up grayscale areas
THRESHOLD = 120  # Increase to make thinner lines; lower for more details

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

        # Binarize image (convert to black & white)
        bw_image = inverted_edges.point(lambda x: 0 if x < THRESHOLD else 255, '1')

        # Save the outlined image
        output_path = os.path.join(output_folder, f"outlined_{filename}")
        bw_image.save(output_path)

print("All images have been outlined and saved to the Outlined folder.")
