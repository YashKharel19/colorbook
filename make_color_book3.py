import cv2
import numpy as np
import os
import re
from tqdm import tqdm
from fpdf import FPDF
from PIL import Image
# import svgwrite # NO LONGER NEEDED
# import cairosvg # NO LONGER NEEDED

# --- Configuration ---
INPUT_FOLDER = "./Nepali"
OUTPUT_PAGES_SUBFOLDER = "pages_raster_outline" # Changed subfolder name
OUTPUT_PDF_SUBFOLDER = "pdf_raster_outline"

OUTPUT_BASE_FOLDER = "ColorBook_Output_Raster"

PDF_OUTPUT_FILENAME = "MyColorBook_RasterOutline.pdf"
PAGE_PREFIX = "page_"

# Outline generation parameters for OpenCV
GAUSSIAN_BLUR_KERNEL_SIZE = (5, 5)
GAUSSIAN_BLUR_SIGMA = 0
CANNY_LOW_THRESHOLD = 50
CANNY_HIGH_THRESHOLD = 150

# Parameters for smoother RASTER outlines
# This upscale factor is applied to the image BEFORE generating the outline.
# The outline will be saved at this higher resolution.
RASTER_OUTLINE_UPSCALE_FACTOR = 2.0 # e.g., 2.0 means 2x resolution for the outline image
RASTER_OUTLINE_THICKNESS = 2      # Thickness of drawn contour lines
RASTER_CONTOUR_RETRIEVAL_MODE = cv2.RETR_LIST

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

# --- RASTER Outline Generation (Replaces SVG version) ---
def create_high_res_raster_outline(image, upscale_factor=1.0, line_thickness=2):
    """
    Generates a smooth, anti-aliased RASTER outline image from the input image
    at a potentially higher resolution.
    """
    original_height, original_width = image.shape[:2]

    # Target dimensions for the outline image
    target_width = int(original_width * upscale_factor)
    target_height = int(original_height * upscale_factor)

    # If upscaling, process on a resized image. Otherwise, use original size.
    if upscale_factor != 1.0:
         # Resize input image for processing contours at higher res
        proc_image_for_contours = cv2.resize(image, (target_width, target_height),
                                             interpolation=cv2.INTER_LANCZOS4)
    else:
        proc_image_for_contours = image.copy()


    gray = cv2.cvtColor(proc_image_for_contours, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, GAUSSIAN_BLUR_KERNEL_SIZE, GAUSSIAN_BLUR_SIGMA)
    edges = cv2.Canny(blurred, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD)
    contours, _ = cv2.findContours(edges, RASTER_CONTOUR_RETRIEVAL_MODE, cv2.CHAIN_APPROX_SIMPLE)

    # Create a blank white canvas at the target (potentially upscaled) resolution
    # It needs to be 3-channel (BGR) for anti-aliased colored lines (black in this case).
    outline_canvas = np.ones((target_height, target_width, 3), dtype=np.uint8) * 255

    # Draw contours with anti-aliasing
    cv2.drawContours(outline_canvas, contours, -1, (0, 0, 0), line_thickness, lineType=cv2.LINE_AA)

    return outline_canvas # This is the high-resolution raster outline image


# --- PDF Generation Function (Simplified for raster outlines) ---
def create_pdf_from_page_assets(page_assets_folder, pdf_output_path):
    original_image_files = sorted(
        [f for f in os.listdir(page_assets_folder) if f.lower().endswith(('_original.png', '_original.jpg'))],
        key=lambda f: get_sort_key_for_filename(f, PAGE_PREFIX)
    )

    if not original_image_files:
        print(f"No original image assets found in '{page_assets_folder}' to create PDF.")
        return

    print(f"\nCreating PDF '{pdf_output_path}' from {len(original_image_files)} pages...")

    pdf = FPDF(orientation='L', unit='mm', format='A4')
    page_width_mm = PDF_PAGE_HEIGHT_MM
    page_height_mm = PDF_PAGE_WIDTH_MM
    usable_width_total = page_width_mm - 2 * PDF_MARGIN_MM
    usable_height_total = page_height_mm - 2 * PDF_MARGIN_MM
    item_usable_width = (usable_width_total - PDF_MARGIN_MM) / 2
    item_usable_height = usable_height_total

    for original_img_filename in tqdm(original_image_files, desc="Adding pages to PDF"):
        base_name = original_img_filename.replace("_original.png", "").replace("_original.jpg", "")
        original_image_path = os.path.join(page_assets_folder, original_img_filename)
        # Expecting raster outline, e.g., page_image1_outline.png
        outline_image_path = os.path.join(page_assets_folder, f"{base_name}_outline.png") # Assuming PNG for outline

        if not os.path.exists(outline_image_path):
            print(f"Warning: Raster outline for {base_name} not found (expected .png). Skipping page.")
            continue

        pdf.add_page()

        try:
            # 1. Add Original Image
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

            # 2. Add High-Res Raster Outline Image
            with Image.open(outline_image_path) as img_outline:
                outline_w_px, outline_h_px = img_outline.size
            aspect_ratio_outline = outline_w_px / outline_h_px
            w_mm_outline = item_usable_width
            h_mm_outline = w_mm_outline / aspect_ratio_outline
            if h_mm_outline > item_usable_height:
                h_mm_outline = item_usable_height
                w_mm_outline = h_mm_outline * aspect_ratio_outline
            x_pos_outline = PDF_MARGIN_MM + item_usable_width + PDF_MARGIN_MM + (item_usable_width - w_mm_outline) / 2
            y_pos_outline = PDF_MARGIN_MM + (item_usable_height - h_mm_outline) / 2
            pdf.image(outline_image_path, x=x_pos_outline, y=y_pos_outline, w=w_mm_outline, h=h_mm_outline)

        except Exception as e:
            print(f"Error adding page for {base_name} to PDF: {e}")
    try:
        pdf.output(pdf_output_path, "F")
        print(f"Successfully created PDF: {pdf_output_path}")
    except Exception as e:
        print(f"Error saving PDF {pdf_output_path}: {e}")

# --- Main Script ---
def generate_color_book_assets_and_pdf():
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
        key=get_sort_key_for_filename
    )
    if not input_image_files:
        print(f"No image files found in '{INPUT_FOLDER}'.")
        return

    print(f"Found {len(input_image_files)} images to process for pages.")
    assets_generated = False

    for filename in tqdm(input_image_files, desc="Generating Page Assets"):
        try:
            input_path = os.path.join(INPUT_FOLDER, filename)
            base_original_name, ext = os.path.splitext(filename)
            page_base_name = f"{PAGE_PREFIX}{base_original_name}"
            
            output_original_image_path = os.path.join(output_pages_dir, f"{page_base_name}_original{ext}")
            # Save raster outline as PNG
            output_raster_outline_path = os.path.join(output_pages_dir, f"{page_base_name}_outline.png")

            original_image = cv2.imread(input_path)
            if original_image is None or original_image.shape[0] == 0 or original_image.shape[1] == 0:
                print(f"Warning: Could not read or empty image {filename}. Skipping.")
                continue

            cv2.imwrite(output_original_image_path, original_image)

            # Create and save the high-resolution RASTER outline
            raster_outline_image = create_high_res_raster_outline(
                original_image,
                upscale_factor=RASTER_OUTLINE_UPSCALE_FACTOR,
                line_thickness=RASTER_OUTLINE_THICKNESS
            )
            cv2.imwrite(output_raster_outline_path, raster_outline_image)
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
        os.makedirs(INPUT_FOLDER)
        print(f"Created dummy '{INPUT_FOLDER}'. Please add images there.")
    generate_color_book_assets_and_pdf()