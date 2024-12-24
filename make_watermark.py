import numpy as np
import pywt
import cv2
import requests
import os
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from scipy.linalg import svd

# Function to embed watermark
def embed_watermark(original_image, watermark_image, alpha=0.1):
    # Read original and watermark images
    original = cv2.imread(original_image, cv2.IMREAD_GRAYSCALE)
    watermark = cv2.imread(watermark_image, cv2.IMREAD_GRAYSCALE)

    if original is None or watermark is None:
        raise ValueError(f"Error loading images. Ensure {original_image} and {watermark_image} exist and are valid images.")

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

    if watermarked is None or original is None:
        raise ValueError(f"Error loading images. Ensure {watermarked_image} and {original_image} exist and are valid images.")

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

# Function to download sample images from SIPI database
def download_sipi_images(base_url, save_dir, num_images=5):
    response = requests.get(base_url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find all image links on the page
    links = [urljoin(base_url, a['href']) for a in soup.find_all('a', href=True) if a['href'].endswith('.tiff')]

    os.makedirs(save_dir, exist_ok=True)
    downloaded_images = []

    for link in links[:num_images]:
        filename = os.path.join(save_dir, os.path.basename(link))
        with requests.get(link, stream=True) as r:
            with open(filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        downloaded_images.append(filename)

    return downloaded_images

# Main code for testing
if __name__ == "__main__":
    # Base URL of SIPI database
    sipi_url = "https://sipi.usc.edu/database/database.php?volume=misc"

    # Directory to save downloaded images
    save_directory = "sipi_images"

    # Download images
    # images = download_sipi_images(sipi_url, save_directory)
    images = [
        r"sipi_images\4.1.01.tiff",
        r"sipi_images\4.1.02.tiff",
        r"sipi_images\4.1.04.tiff",
        # Add more image paths as needed
    ]

    # Assuming a local watermark image is available
    watermark_image = "watermark_image1.png"

    # Process each downloaded image
    for image in images:
        try:
            print(f"Processing {image}...")
            # Embed watermark
            watermarked = embed_watermark(image, watermark_image, alpha=0.1)
            watermarked_filename = f"{os.path.splitext(image)[0]}_watermarked.png"
            cv2.imwrite(watermarked_filename, watermarked)

            # Extract watermark
            extracted = extract_watermark(watermarked_filename, image, alpha=0.1)
            extracted_filename = f"{os.path.splitext(image)[0]}_extracted_watermark.png"
            cv2.imwrite(extracted_filename, extracted)

            print(f"Processed {image}: Watermarked saved to {watermarked_filename}, Extracted watermark saved to {extracted_filename}")
        except Exception as e:
            print("error")
            print(f"Error processing {image}: {e}")
    