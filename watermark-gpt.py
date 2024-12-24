import numpy as np
import pywt
import cv2
import os
from scipy.linalg import svd

# Function to embed watermark
def embed_watermark(original_image, watermark_image1, alpha=0.1):
    # Read original and watermark images
    original = cv2.imread(original_image, cv2.IMREAD_GRAYSCALE)
    watermark = cv2.imread(watermark_image1, cv2.IMREAD_GRAYSCALE)

    # Resize watermark to match the original image dimensions
    watermark_resized = cv2.resize(watermark, (original.shape[1], original.shape[0]))

    # Apply DWT on original image
    coeffs = pywt.dwt2(original, 'haar')
    LL, (LH, HL, HH) = coeffs

    # Perform SVD on the LL sub-band
    U, S, Vt = svd(LL, full_matrices=False)

    # Add watermark to singular values
    watermark_flat = watermark_resized.flatten()
    S_marked = S + alpha * watermark_flat[:len(S)]

    # Reconstruct the LL sub-band with modified singular values
    LL_marked = np.dot(U, np.dot(np.diag(S_marked), Vt))

    # Combine modified LL with other sub-bands
    coeffs_marked = (LL_marked, (LH, HL, HH))
    watermarked_image = pywt.idwt2(coeffs_marked, 'haar')

    return np.uint8(np.clip(watermarked_image, 0, 255))

# Function to extract watermark
def extract_watermark(watermarked_image, original_image, alpha=0.1):
    # Read watermarked and original images
    watermarked = cv2.imread(watermarked_image, cv2.IMREAD_GRAYSCALE)
    original = cv2.imread(original_image, cv2.IMREAD_GRAYSCALE)

    # Apply DWT on both images
    coeffs_wm = pywt.dwt2(watermarked, 'haar')
    coeffs_orig = pywt.dwt2(original, 'haar')

    LL_wm, _ = coeffs_wm
    LL_orig, _ = coeffs_orig

    # Perform SVD on both LL sub-bands
    Uw, Sw, Vtw = svd(LL_wm, full_matrices=False)
    Uo, So, Vto = svd(LL_orig, full_matrices=False)

    # Extract watermark from the singular values
    watermark_flat = (Sw - So) / alpha
    watermark = watermark_flat.reshape((original.shape[0], original.shape[1]))

    return np.uint8(np.clip(watermark, 0, 255))

# Main code for testing
if __name__ == "__main__":
    # List of manually downloaded images
    images = [
        r"sipi_images\4.1.01.tiff",
        r"sipi_images\4.1.02.tiff",
        r"sipi_images\4.1.04.tiff",
        # Add more image paths as needed
    ]

    # Assuming a local watermark image is available
    watermark_image1 = "watermark_image1.png"

    # Process each downloaded image
    for image in images:
        # Embed watermark
        watermarked = embed_watermark(image, watermark_image1, alpha=0.1)
        watermarked_filename = f"{os.path.splitext(image)[0]}_watermarked.png"
        cv2.imwrite(watermarked_filename, watermarked)

        # Extract watermark
        extracted = extract_watermark(watermarked_filename, image, alpha=0.1)
        extracted_filename = f"{os.path.splitext(image)[0]}_extracted_watermark.png"
        cv2.imwrite(extracted_filename, extracted)

        print(f"Processed {image}: Watermarked saved to {watermarked_filename}, Extracted watermark saved to {extracted_filename}")