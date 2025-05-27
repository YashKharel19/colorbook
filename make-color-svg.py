import cv2
import numpy as np
import os
import re
from tqdm import tqdm
from fpdf import FPDF
from PIL import Image
import svgwrite # For creating SVG files
import cairosvg # For converting SVG to PNG for PDF

# --- Configuration ---
INPUT_FOLDER = "images_folder"
OUTPUT_PAGES_SUBFOLDER = "pages" # Subfolder within OUTPUT_FOLDER for individual page components
OUTPUT_PDF_SUBFOLDER = "pdf"     # Subfolder within OUTPUT_FOLDER for final PDF

# Base output folder
OUTPUT_BASE_FOLDER = "ColorBook_Output"

# PDF_OUTPUT_FILENAME will now be placed inside OUTPUT_BASE_FOLDER/OUTPUT_PDF_SUBFOLDER
PDF_OUTPUT_FILENAME = "MyColorBook_VectorOutline.pdf"
PAGE_PREFIX = "page_"

# Outline generation parameters
GAUSSIAN_BLUR_KERNEL_SIZE = (5, 5)
GAUSSIAN_BLUR_SIGMA = 0
CANNY_LOW_THRESHOLD = 50
CANNY_HIGH_THRESHOLD = 150

# Parameters for smoother vector outlines
# This upscale factor is for the image processed by Canny to get better contours
# The SVG itself will be defined by these contour points.
CONTOUR_UPSCALE_FACTOR = 1.5 # Increase for more detailed contours (can be 1.0 for no upscale)
SVG_STROKE_WIDTH = 1.5       # Stroke width for lines in the SVG

CONTOUR_RETRIEVAL_MODE = cv2.RETR_LIST
# For PDF page generation from SVG
SVG_RASTER_DPI = 300 # DPI for converting SVG to PNG for PDF inclusion

# PDF Page settings
PDF_PAGE_WIDTH_MM = 210
PDF_PAGE_HEIGHT_MM = 297
PDF_MARGIN_MM = 10

# --- Sorting Helper Functions (same as before) ---
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def get_sort_key_for_filename(filename, prefix_to_strip=""):
    try:
        name_without_prefix = filename
        if prefix_to_strip and filename.lower().startswith(prefix_to_strip.lower()):
            name_without_prefix = filename[len(prefix_to_strip):]
        base_original = os.path.splitext(name_without_prefix)[0]
        return natural_sort_key(base_original)
    except Exception:
        return [filename.lower()]

# --- Image Processing & SVG Outline Generation ---
def create_vector_outline_svg(image, output_svg_path,
                              upscale_factor=1.0, stroke_width=1.0):
    """
    Generates a vector outline (SVG) from the input image.
    """
    original_height, original_width = image.shape[:2]

    if upscale_factor > 1.0:
        proc_image = cv2.resize(image,
                                (int(original_width * upscale_factor), int(original_height * upscale_factor)),
                                interpolation=cv2.INTER_LANCZOS4)
    else:
        proc_image = image.copy()

    # Dimensions of the image on which contours are found (this will be SVG viewBox)
    svg_viewbox_width = proc_image.shape[1]
    svg_viewbox_height = proc_image.shape[0]

    gray = cv2.cvtColor(proc_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, GAUSSIAN_BLUR_KERNEL_SIZE, GAUSSIAN_BLUR_SIGMA)
    edges = cv2.Canny(blurred, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD)
    contours, _ = cv2.findContours(edges, CONTOUR_RETRIEVAL_MODE, cv2.CHAIN_APPROX_SIMPLE)

    # Create SVG drawing
    dwg = svgwrite.Drawing(output_svg_path,
                           size=(f"{svg_viewbox_width}px", f"{svg_viewbox_height}px"), # Can be omitted if viewBox is set
                           viewBox=(f"0 0 {svg_viewbox_width} {svg_viewbox_height}"),
                           profile='tiny') # 'tiny' is a common profile

    # Add contours as paths to SVG
    # Scale stroke width if we upscaled, so it appears correct when SVG is rendered at original size
    # However, SVG stroke is unitless in context of viewBox, so direct stroke_width is usually fine.
    # Effective stroke width will depend on how SVG is scaled when rendered.
    # For simplicity, we use the provided stroke_width directly.
    for contour in contours:
        if len(contour) > 1: # Need at least 2 points for a line
            points = []
            for point_outer_array in contour:
                points.append(tuple(point_outer_array[0])) # Extract (x,y)

            # Create a polyline for open contours or path for closed (polyline is simpler here)
            # Using path for more flexibility if we wanted curves later.
            # For simple lines from Canny, polyline is fine.
            # path_data = "M {} {}".format(points[0][0], points[0][1])
            # for pt in points[1:]:
            #     path_data += " L {} {}".format(pt[0], pt[1])
            # dwg.add(dwg.path(d=path_data, stroke='black', fill='none', stroke_width=stroke_width))
            dwg.add(dwg.polyline(points, stroke='black', fill='none', stroke_width=stroke_width))


    dwg.save()
    return svg_viewbox_width, svg_viewbox_height # Return dimensions for PDF rasterization


# --- PDF Generation Function (Modified for SVG handling) ---
def create_pdf_from_page_assets(page_assets_folder, pdf_output_path):
    """
    Creates a PDF. Each page in the PDF will show:
    1. The original color image (raster).
    2. The vector outline (rendered from SVG to high-res PNG).
    These are placed side-by-side.
    """
    # Find all original images, assuming they indicate a page
    original_image_files = sorted(
        [f for f in os.listdir(page_assets_folder) if f.lower().endswith(('_original.png', '_original.jpg'))],
        key=lambda f: get_sort_key_for_filename(f, PAGE_PREFIX) # Sort by base name after prefix
    )

    if not original_image_files:
        print(f"No original image assets found in '{page_assets_folder}' to create PDF.")
        return

    print(f"\nCreating PDF '{pdf_output_path}' from {len(original_image_files)} pages...")

    pdf = FPDF(orientation='L', unit='mm', format='A4') # LANDSCAPE to better fit two images
    page_width_mm = PDF_PAGE_HEIGHT_MM # Swapped for landscape
    page_height_mm = PDF_PAGE_WIDTH_MM # Swapped for landscape
    usable_width_total = page_width_mm - 2 * PDF_MARGIN_MM
    usable_height_total = page_height_mm - 2 * PDF_MARGIN_MM

    # Each item (original or outline) will get half the usable width
    item_usable_width = (usable_width_total - PDF_MARGIN_MM) / 2 # -PDF_MARGIN_MM for gap
    item_usable_height = usable_height_total

    temp_outline_png_path = os.path.join(page_assets_folder, "_temp_outline.png")

    for original_img_filename in tqdm(original_image_files, desc="Adding pages to PDF"):
        base_name = original_img_filename.replace("_original.png", "").replace("_original.jpg", "")
        original_image_path = os.path.join(page_assets_folder, original_img_filename)
        outline_svg_path = os.path.join(page_assets_folder, f"{base_name}_outline.svg")

        if not os.path.exists(outline_svg_path):
            print(f"Warning: SVG outline for {base_name} not found. Skipping page.")
            continue

        pdf.add_page()

        try:
            # 1. Add Original Image (Raster)
            with Image.open(original_image_path) as img:
                orig_w_px, orig_h_px = img.size

            aspect_ratio_orig = orig_w_px / orig_h_px
            w_mm_orig = item_usable_width
            h_mm_orig = w_mm_orig / aspect_ratio_orig
            if h_mm_orig > item_usable_height:
                h_mm_orig = item_usable_height
                w_mm_orig = h_mm_orig * aspect_ratio_orig

            x_pos_orig = PDF_MARGIN_MM + (item_usable_width - w_mm_orig) / 2
            y_pos_orig = PDF_MARGIN_MM + (item_usable_height - h_mm_orig) / 2
            pdf.image(original_image_path, x=x_pos_orig, y=y_pos_orig, w=w_mm_orig, h=h_mm_orig)

            # 2. Convert SVG to High-Res PNG and Add
            # Get SVG intrinsic dimensions from its viewBox (if needed, or assume from original)
            # For robust DPI conversion, cairosvg needs width/height or scale
            # We can infer target pixel width from SVG's aspect ratio and desired PDF print size
            svg_target_width_px = int((item_usable_width / 25.4) * SVG_RASTER_DPI) # mm to inches * DPI

            cairosvg.svg2png(url=outline_svg_path, write_to=temp_outline_png_path,
                             output_width=svg_target_width_px, dpi=SVG_RASTER_DPI)
                             # parent_width, parent_height can also be used for scaling

            with Image.open(temp_outline_png_path) as img_outline:
                outline_w_px, outline_h_px = img_outline.size

            aspect_ratio_outline = outline_w_px / outline_h_px
            w_mm_outline = item_usable_width
            h_mm_outline = w_mm_outline / aspect_ratio_outline
            if h_mm_outline > item_usable_height:
                h_mm_outline = item_usable_height
                w_mm_outline = h_mm_outline * aspect_ratio_outline

            x_pos_outline = PDF_MARGIN_MM + item_usable_width + PDF_MARGIN_MM + (item_usable_width - w_mm_outline) / 2
            y_pos_outline = PDF_MARGIN_MM + (item_usable_height - h_mm_outline) / 2
            pdf.image(temp_outline_png_path, x=x_pos_outline, y=y_pos_outline, w=w_mm_outline, h=h_mm_outline)

        except Exception as e:
            print(f"Error adding page for {base_name} to PDF: {e}")

    if os.path.exists(temp_outline_png_path):
        try:
            os.remove(temp_outline_png_path)
        except Exception as e:
            print(f"Warning: Could not remove temporary file {temp_outline_png_path}: {e}")
    try:
        pdf.output(pdf_output_path, "F")
        print(f"Successfully created PDF: {pdf_output_path}")
    except Exception as e:
        print(f"Error saving PDF {pdf_output_path}: {e}")


# --- Main Script ---
def generate_color_book_assets_and_pdf():
    # Create base output structure
    # Output structure:
    # ColorBook_Output/
    #   pages/  <-- individual original images and SVG outlines
    #     page_image1_original.png
    #     page_image1_outline.svg
    #     page_image2_original.png
    #     page_image2_outline.svg
    #     ...
    #   pdf/    <-- final PDF
    #     MyColorBook_VectorOutline.pdf

    output_pages_dir = os.path.join(OUTPUT_BASE_FOLDER, OUTPUT_PAGES_SUBFOLDER)
    output_pdf_dir = os.path.join(OUTPUT_BASE_FOLDER, OUTPUT_PDF_SUBFOLDER)

    os.makedirs(output_pages_dir, exist_ok=True)
    os.makedirs(output_pdf_dir, exist_ok=True)

    final_pdf_path = os.path.join(output_pdf_dir, PDF_OUTPUT_FILENAME)

    if not os.path.exists(INPUT_FOLDER):
        print(f"Error: Input folder '{INPUT_FOLDER}' not found.")
        return

    print(f"Page assets will be saved in '{output_pages_dir}'")
    print(f"Final PDF will be saved as '{final_pdf_path}'")

    input_image_files = sorted(
        [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'))],
        key=get_sort_key_for_filename # Sort by base name
    )

    if not input_image_files:
        print(f"No image files found in '{INPUT_FOLDER}'.")
        return

    print(f"Found {len(input_image_files)} images to process for pages.")
    assets_generated = False

    for filename in tqdm(input_image_files, desc="Generating Page Assets"):
        try:
            input_path = os.path.join(INPUT_FOLDER, filename)
            base_original_name, ext = os.path.splitext(filename) # e.g. "image1", ".png"
            
            # Define output asset names
            page_base_name = f"{PAGE_PREFIX}{base_original_name}" # e.g. "page_image1"
            
            output_original_image_path = os.path.join(output_pages_dir, f"{page_base_name}_original{ext}")
            output_svg_outline_path = os.path.join(output_pages_dir, f"{page_base_name}_outline.svg")

            original_image = cv2.imread(input_path)
            if original_image is None:
                print(f"Warning: Could not read image {filename}. Skipping.")
                continue
            if original_image.shape[0] == 0 or original_image.shape[1] == 0:
                print(f"Warning: Image {filename} has zero dimension. Skipping.")
                continue

            # Save a copy of the original image (or just use input_path if no modification)
            cv2.imwrite(output_original_image_path, original_image)

            # Create and save the vector outline SVG
            create_vector_outline_svg(original_image, output_svg_outline_path,
                                      upscale_factor=CONTOUR_UPSCALE_FACTOR,
                                      stroke_width=SVG_STROKE_WIDTH)
            assets_generated = True

        except Exception as e:
            print(f"Error processing {filename} for assets: {e}")

    print("\nIndividual page asset generation complete!")

    if assets_generated:
        create_pdf_from_page_assets(output_pages_dir, final_pdf_path)
    else:
        print("No page assets were generated, so PDF creation is skipped.")

    print(f"\nProcess finished. Check '{output_pages_dir}' for assets and '{final_pdf_path}' for the PDF.")

if __name__ == "__main__":
    if not os.path.exists(INPUT_FOLDER):
        print(f"Creating a dummy '{INPUT_FOLDER}' for testing. Please add your images there.")
        os.makedirs(INPUT_FOLDER)
        if not os.listdir(INPUT_FOLDER):
            for i in [1, 10, 2]:
                 dummy_img = np.zeros((300,400,3), dtype=np.uint8)
                 cv2.circle(dummy_img, (100,100), 80, (0,255,0), -1)
                 cv2.rectangle(dummy_img, (200,150), (350,250), (0,0,255), 3)
                 cv2.putText(dummy_img, f"Img {i}", (50,250), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)
                 cv2.imwrite(os.path.join(INPUT_FOLDER, f"sample_image_{i}.png"), dummy_img)
            print(f"Added sample images to '{INPUT_FOLDER}' for testing.")

    generate_color_book_assets_and_pdf()