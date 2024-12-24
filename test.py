from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np
import os

def compare_images(image1_path, image2_path):
    img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        print(f"Error: One of the images {image1_path} or {image2_path} could not be loaded.")
        return

    # Compute SSIM
    similarity, _ = ssim(img1, img2, full=True)
    print(f"SSIM between {image1_path} and {image2_path}: {similarity:.4f}")

    # Compute PSNR
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        psnr = 100  # Images are identical
    else:
        pixel_max = 255.0
        psnr = 20 * np.log10(pixel_max / np.sqrt(mse))
    print(f"PSNR between {image1_path} and {image2_path}: {psnr:.2f} dB")

# Main code for testing
if __name__ == "__main__":
    # Directory containing images
    image_dir = "sipi_images"

    # List of images to compare
    images = [
        "4.1.01.tiff",
        "4.1.02.tiff",
        "4.1.04.tiff",
        # Add more image filenames as needed
    ]

    # Compare original images with watermarked images
    for image in images:
        original_image_path = os.path.join(image_dir, image)
        watermarked_image_path = os.path.join(image_dir, f"{os.path.splitext(image)[0]}_watermarked.png")
        extracted_watermark_path = os.path.join(image_dir, f"{os.path.splitext(image)[0]}_extracted_watermark.png")

        print(f"Comparing original image with watermarked image for {image}...")
        compare_images(original_image_path, watermarked_image_path)

        # Assuming watermark_image1.png is the original watermark image
        original_watermark_path = "watermark_image1.png"

        # Check if the extracted watermark file exists
        if not os.path.exists(extracted_watermark_path):
            print(f"Error: Extracted watermark image not found: {extracted_watermark_path}")
            continue

        print(f"Comparing original watermark with extracted watermark for {image}...")
        compare_images(original_watermark_path, extracted_watermark_path)