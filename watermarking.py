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

def compass_edge_detector(img_block):
    """Applies a compass edge detector to an image block."""
    h, w = img_block.shape
    edges = np.zeros_like(img_block)
    kernels = [
        np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]),  # E
        np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),  # S
        np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]),  # W
        np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]),  # N
        np.array([[0, 1, 1], [-1, 0, 1], [-1, -1, 0]]), # SE
        np.array([[-1, -1, 0], [-1, 0, 1], [0, 1, 1]]), # NE
        np.array([[0, -1, -1], [1, 0, -1], [1, 1, 0]]), # NW
        np.array([[1, 1, 0], [1, 0, -1], [0, -1, -1]])  # SW
    ]

    max_magnitude = np.zeros_like(img_block, dtype=float)
    for kernel in kernels:
        filtered_block = cv2.filter2D(img_block, -1, kernel)
        magnitude = np.abs(filtered_block)
        max_magnitude = np.maximum(max_magnitude, magnitude)

    return max_magnitude

def embed_watermark(host_image_path, watermark_image_path, embedded_image_path):
    """Embeds the watermark into the host image using compass edge detection and LSB."""
    host_img = cv2.imread(host_image_path, cv2.IMREAD_GRAYSCALE)
    watermark_img = cv2.imread(watermark_image_path, cv2.IMREAD_GRAYSCALE)

    # Resize watermark if needed
    watermark_img = cv2.resize(watermark_img, (32, 32))  # Assuming watermark size is 32x32
    watermark_img = watermark_img > 128 # Convert to binary image (0 or 255 -> True or False)

    host_h, host_w = host_img.shape
    watermark_h, watermark_w = watermark_img.shape
    block_size = 16

    embedded_img = host_img.copy()

    for i in range(0, host_h - block_size + 1, block_size):
        for j in range(0, host_w - block_size + 1, block_size):
            host_block = host_img[i:i + block_size, j:j + block_size]
            watermark_block = watermark_img[
                             (i // block_size) * (watermark_h // block_size): (i // block_size + 1) * (watermark_h // block_size),
                             (j // block_size) * (watermark_w // block_size): (j // block_size + 1) * (watermark_w // block_size)
                         ]

            if watermark_block.shape[0] == 0 or watermark_block.shape[1] == 0:
                continue

            edge_magnitude = compass_edge_detector(host_block)
            max_edge = np.max(edge_magnitude)

            # Reshape watermark block to match host block size
            resized_watermark_block = cv2.resize(watermark_block.astype(np.uint8) * 1, (block_size, block_size), interpolation=cv2.INTER_NEAREST)

            for row in range(block_size):
                for col in range(block_size):
                    host_pixel = bin(embedded_img[i + row, j + col])[2:].zfill(8)
                    watermark_bit = int(resized_watermark_block[row, col])

                    if edge_magnitude[row, col] >= max_edge / 2:
                        # Alter 1 LSB
                        embedded_pixel = host_pixel[:-1] + str(watermark_bit)
                    else:
                        # Alter 2 LSBs
                        embedded_pixel = host_pixel[:-2] + bin(watermark_bit)[2:].zfill(2)[-2:]

                    embedded_img[i + row, j + col] = int(embedded_pixel, 2)

    cv2.imwrite(embedded_image_path, embedded_img)
    print(f"Watermark embedded successfully in {embedded_image_path}")

def extract_watermark(embedded_image_path, original_image_path, extracted_watermark_path):
    """Extracts the watermark from the embedded image."""
    embedded_img = cv2.imread(embedded_image_path, cv2.IMREAD_GRAYSCALE)
    original_img = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)

    host_h, host_w = embedded_img.shape
    block_size = 16
    extracted_watermark = np.zeros((host_h // block_size * 32 // block_size, host_w // block_size * 32 // block_size), dtype=np.uint8) # Assuming original watermark was 32x32

    for i in range(0, host_h - block_size + 1, block_size):
        for j in range(0, host_w - block_size + 1, block_size):
            embedded_block = embedded_img[i:i + block_size, j:j + block_size]
            original_block = original_img[i:i + block_size, j:j + block_size]

            edge_magnitude = compass_edge_detector(original_block)
            max_edge = np.max(edge_magnitude)

            extracted_watermark_block = np.zeros((block_size, block_size), dtype=int)

            for row in range(block_size):
                for col in range(block_size):
                    embedded_pixel = bin(embedded_block[row, col])[2:].zfill(8)

                    if edge_magnitude[row, col] >= max_edge / 2:
                        # Extract 1 LSB
                        watermark_bit = int(embedded_pixel[-1])
                    else:
                        # Extract 2 LSBs
                        watermark_bit = int(embedded_pixel[-2:], 2)

                    extracted_watermark_block[row, col] = watermark_bit

            # Place the extracted block into the larger watermark image
            wm_i_start = (i // block_size) * (32 // block_size)
            wm_i_end = wm_i_start + 1
            wm_j_start = (j // block_size) * (32 // block_size)
            wm_j_end = wm_j_start + 1

            extracted_watermark[wm_i_start, wm_j_start] = extracted_watermark_block[0,0] * 255 # Assuming watermark block is binary

    cv2.imwrite(extracted_watermark_path, extracted_watermark)
    print(f"Watermark extracted successfully to {extracted_watermark_path}")

# Main code for testing
if __name__ == "__main__":
    image_dir = "sipi_images"
    images = [
        "4.1.01.tiff",
        "4.1.02.tiff",
        "4.1.04.tiff",
    ]
    watermark_image = "watermark_image1.png"

    for image in images:
        original_image_path = os.path.join(image_dir, image)
        watermarked_image_path = os.path.join(image_dir, f"{os.path.splitext(image)[0]}_watermarked.png")
        extracted_watermark_path = os.path.join(image_dir, f"{os.path.splitext(image)[0]}_extracted_watermark.png")

        # Embed watermark
        embed_watermark(original_image_path, watermark_image, watermarked_image_path)

        # Extract watermark
        extract_watermark(watermarked_image_path, original_image_path, extracted_watermark_path)

        # Compare original image with watermarked image
        print(f"Comparing original image with watermarked image for {image}...")
        compare_images(original_image_path, watermarked_image_path)

        # Check if the extracted watermark file exists
        if not os.path.exists(extracted_watermark_path):
            print(f"Error: Extracted watermark image not found: {extracted_watermark_path}")
            continue

        # Compare original watermark with extracted watermark
        print(f"Comparing original watermark with extracted watermark for {image}...")
        compare_images(watermark_image, extracted_watermark_path)