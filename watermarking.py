# import cv2
# import numpy as np
# from PIL import Image
# import random

# def generate_s_box(p, x0, size=256):
#     """Generates an S-box using the Piecewise Linear Chaotic Map (PWLCM)."""
#     s_box = list(range(size))
#     chaos_sequence = []
#     x = x0
#     for _ in range(size * 100):  # Iterate to achieve better randomness
#         if 0 <= x < p:
#             x_next = x / p
#         elif p <= x < 0.5:
#             x_next = (0.5 - x) / (0.5 - p)
#         elif 0.5 <= x < 1 - p:
#             x_next = (x - 0.5) / (0.5 - p)
#         elif 1 - p <= x < 1:
#             x_next = (1 - x) / p
#         chaos_sequence.append(x_next)
#         x = x_next

#     # Select a portion of the sequence and map to the S-box
#     start_index = size * 50 # Start after some iterations
#     relevant_sequence = chaos_sequence[start_index:start_index + size]

#     # Sort the sequence and map indices to S-box values
#     sorted_indices = sorted(range(size), key=lambda k: relevant_sequence[k])
#     permuted_s_box = [0] * size
#     for i, index in enumerate(sorted_indices):
#         permuted_s_box[index] = i

#     return permuted_s_box

# def encrypt_watermark(watermark_block, s_box):
#     """Encrypts the watermark block using the S-box."""
#     encrypted_block = np.zeros_like(watermark_block)
#     for i in range(watermark_block.shape[0]):
#         for j in range(watermark_block.shape[1]):
#             val = int(watermark_block[i, j])
#             encrypted_block[i, j] = s_box[val]
#     return encrypted_block

# def decrypt_watermark(encrypted_block, s_box):
#     """Decrypts the watermark block using the S-box."""
#     decrypted_block = np.zeros_like(encrypted_block)
#     inverse_s_box = {v: k for k, v in enumerate(s_box)}
#     for i in range(encrypted_block.shape[0]):
#         for j in range(encrypted_block.shape[1]):
#             val = int(encrypted_block[i, j])
#             decrypted_block[i, j] = inverse_s_box[val]
#     return decrypted_block

# def compass_edge_detector(img_block):
#     """Applies a compass edge detector to an image block."""
#     h, w = img_block.shape
#     edges = np.zeros_like(img_block)
#     kernels = [
#         np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]),  # E
#         np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),  # S
#         np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]),  # W
#         np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]),  # N
#         np.array([[0, 1, 1], [-1, 0, 1], [-1, -1, 0]]), # SE
#         np.array([[-1, -1, 0], [-1, 0, 1], [0, 1, 1]]), # NE
#         np.array([[0, -1, -1], [1, 0, -1], [1, 1, 0]]), # NW
#         np.array([[1, 1, 0], [1, 0, -1], [0, -1, -1]])  # SW
#     ]

#     max_magnitude = np.zeros_like(img_block, dtype=float)
#     for kernel in kernels:
#         filtered_block = cv2.filter2D(img_block, -1, kernel)
#         magnitude = np.abs(filtered_block)
#         max_magnitude = np.maximum(max_magnitude, magnitude)

#     return max_magnitude

# def embed_watermark(host_image_path, watermark_image_path, embedded_image_path, p, x0):
#     """Embeds the watermark into the host image using compass edge detection and LSB."""
#     host_img = cv2.imread(host_image_path, cv2.IMREAD_GRAYSCALE)
#     watermark_img = cv2.imread(watermark_image_path, cv2.IMREAD_GRAYSCALE)

#     # Resize watermark if needed
#     watermark_img = cv2.resize(watermark_img, (32, 32))  # Assuming watermark size is 32x32
#     watermark_img = watermark_img > 128 # Convert to binary image (0 or 255 -> 0 or 1)

#     s_box = generate_s_box(p, x0)

#     host_h, host_w = host_img.shape
#     watermark_h, watermark_w = watermark_img.shape
#     block_size = 16

#     embedded_img = host_img.copy()

#     for i in range(0, host_h - block_size + 1, block_size):
#         for j in range(0, host_w - block_size + 1, block_size):
#             host_block = host_img[i:i + block_size, j:j + block_size]
#             watermark_block = watermark_img[
#                              (i // block_size) * (watermark_h // block_size): (i // block_size + 1) * (watermark_h // block_size),
#                              (j // block_size) * (watermark_w // block_size): (j // block_size + 1) * (watermark_w // block_size)
#                          ]

#             if watermark_block.shape[0] == 0 or watermark_block.shape[1] == 0:
#                 continue

#             edge_magnitude = compass_edge_detector(host_block)
#             max_edge = np.max(edge_magnitude)

#             # Reshape watermark block to match host block size
#             resized_watermark_block = cv2.resize(watermark_block.astype(np.uint8) * 255, (block_size, block_size), interpolation=cv2.INTER_NEAREST)
#             resized_watermark_block = resized_watermark_block > 128 # Ensure binary

#             encrypted_watermark_block = encrypt_watermark(resized_watermark_block.astype(int), s_box)

#             for row in range(block_size):
#                 for col in range(block_size):
#                     host_pixel = bin(embedded_img[i + row, j + col])[2:].zfill(8)
#                     watermark_bit = int(encrypted_watermark_block[row, col])

#                     if edge_magnitude[row, col] >= max_edge / 2:
#                         # Alter 1 LSB
#                         embedded_pixel = host_pixel[:-1] + str(watermark_bit)
#                     else:
#                         # Alter 2 LSBs
#                         embedded_pixel = host_pixel[:-2] + bin(watermark_bit)[2:].zfill(2)[-2:] # Take last 2 bits

#                     embedded_img[i + row, j + col] = int(embedded_pixel, 2)

#     cv2.imwrite(embedded_image_path, embedded_img)
#     print(f"Watermark embedded successfully in {embedded_image_path}")

# def extract_watermark(embedded_image_path, original_image_path, extracted_watermark_path, p, x0):
#     """Extracts the watermark from the embedded image."""
#     embedded_img = cv2.imread(embedded_image_path, cv2.IMREAD_GRAYSCALE)
#     original_img = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)

#     s_box = generate_s_box(p, x0)

#     host_h, host_w = embedded_img.shape
#     block_size = 16
#     extracted_watermark = np.zeros((host_h // block_size * 32, host_w // block_size * 32), dtype=np.uint8) # Assuming original watermark was 32x32 per block

#     for i in range(0, host_h - block_size + 1, block_size):
#         for j in range(0, host_w - block_size + 1, block_size):
#             embedded_block = embedded_img[i:i + block_size, j:j + block_size]
#             original_block = original_img[i:i + block_size, j:j + block_size]

#             edge_magnitude = compass_edge_detector(original_block)
#             max_edge = np.max(edge_magnitude)

#             extracted_watermark_block_encrypted = np.zeros((block_size, block_size), dtype=int)

#             for row in range(block_size):
#                 for col in range(block_size):
#                     embedded_pixel = bin(embedded_block[row, col])[2:].zfill(8)

#                     if edge_magnitude[row, col] >= max_edge / 2:
#                         # Extract 1 LSB
#                         watermark_bit = int(embedded_pixel[-1])
#                     else:
#                         # Extract 2 LSBs
#                         watermark_bit = int(embedded_pixel[-2:], 2)

#                     extracted_watermark_block_encrypted[row, col] = watermark_bit

#             decrypted_watermark_block = decrypt_watermark(extracted_watermark_block_encrypted, s_box)

#             # Place the extracted block into the larger watermark image
#             wm_i_start = (i // block_size) * (32 // block_size) * block_size
#             wm_i_end = wm_i_start + block_size
#             wm_j_start = (j // block_size) * (32 // block_size) * block_size
#             wm_j_end = wm_j_start + block_size

#             extracted_watermark[wm_i_start:wm_i_end, wm_j_start:wm_j_end] = decrypted_watermark_block.astype(np.uint8) * 255

#     cv2.imwrite(extracted_watermark_path, extracted_watermark)
#     print(f"Watermark extracted successfully to {extracted_watermark_path}")

# def verify_watermark(original_watermark_path, extracted_watermark_path):
#     """Verifies the extracted watermark against the original."""
#     original_wm = cv2.imread(original_watermark_path, cv2.IMREAD_GRAYSCALE)
#     extracted_wm = cv2.imread(extracted_watermark_path, cv2.IMREAD_GRAYSCALE)

#     if original_wm is None or extracted_wm is None:
#         print("Error: Could not load watermark images for verification.")
#         return

#     # Resize extracted watermark to match original size
#     extracted_wm_resized = cv2.resize(extracted_wm, original_wm.shape[::-1], interpolation=cv2.INTER_NEAREST)

#     if original_wm.shape != extracted_wm_resized.shape:
#         print("Error: Watermark shapes do not match for verification.")
#         return

#     difference = cv2.absdiff(original_wm, extracted_wm_resized)
#     _, thresholded_diff = cv2.threshold(difference, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#     non_zero_pixels = np.count_nonzero(thresholded_diff)
#     total_pixels = original_wm.size
#     similarity = (total_pixels - non_zero_pixels) / total_pixels * 100

#     print(f"Watermark Verification:")
#     print(f"  Similarity: {similarity:.2f}%")

# if __name__ == "__main__":
#     host_image_path = "lena.png"  # Replace with your host image path
#     watermark_image_path = "watermark.png"  # Replace with your watermark image path (should be small and binary)
#     embedded_image_path = "embedded_image.png"
#     extracted_watermark_path = "extracted_watermark.png"

#     # PWLCM parameters
#     p = 0.4
#     x0 = 0.7

#     # Embed the watermark
#     embed_watermark(host_image_path, watermark_image_path, embedded_image_path, p, x0)

#     # Extract the watermark
#     extract_watermark(embedded_image_path, host_image_path, extracted_watermark_path, p, x0)

#     # Verify the extracted watermark
#     verify_watermark(watermark_image_path, extracted_watermark_path)



import cv2
import numpy as np
from PIL import Image
import random

def generate_s_box(p, x0, size=256):
    """Generates an S-box using the Piecewise Linear Chaotic Map (PWLCM)."""
    s_box = list(range(size))
    chaos_sequence = []
    x = x0
    for _ in range(size * 100):  # Iterate to achieve better randomness
        if 0 <= x < p:
            x_next = x / p
        elif p <= x < 0.5:
            x_next = (0.5 - x) / (0.5 - p)
        elif 0.5 <= x < 1 - p:
            x_next = (x - 0.5) / (0.5 - p)
        elif 1 - p <= x < 1:
            x_next = (1 - x) / p
        chaos_sequence.append(x_next)
        x = x_next

    # Select a portion of the sequence and map to the S-box
    start_index = size * 50 # Start after some iterations
    relevant_sequence = chaos_sequence[start_index:start_index + size]

    # Sort the sequence and map indices to S-box values
    sorted_indices = sorted(range(size), key=lambda k: relevant_sequence[k])
    permuted_s_box = [0] * size
    for i, index in enumerate(sorted_indices):
        permuted_s_box[index] = i

    return permuted_s_box

def encrypt_watermark(watermark_block, s_box):
    """Encrypts the watermark block using the S-box."""
    encrypted_block = np.zeros_like(watermark_block)
    for i in range(watermark_block.shape[0]):
        for j in range(watermark_block.shape[1]):
            val = int(watermark_block[i, j])
            encrypted_block[i, j] = s_box[val]
    return encrypted_block

def decrypt_watermark(encrypted_block, s_box):
    """Decrypts the watermark block using the S-box."""
    decrypted_block = np.zeros_like(encrypted_block)
    inverse_s_box = {v: k for k, v in enumerate(s_box)}
    for i in range(encrypted_block.shape[0]):
        for j in range(encrypted_block.shape[1]):
            val = int(encrypted_block[i, j])
            decrypted_block[i, j] = inverse_s_box[val]
    return decrypted_block

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

def embed_watermark(host_image_path, watermark_image_path, embedded_image_path, p, x0):
    """Embeds the watermark into the host image using compass edge detection and LSB."""
    host_img = cv2.imread(host_image_path, cv2.IMREAD_GRAYSCALE)
    watermark_img = cv2.imread(watermark_image_path, cv2.IMREAD_GRAYSCALE)

    # Resize watermark if needed
    watermark_img = cv2.resize(watermark_img, (32, 32))  # Assuming watermark size is 32x32
    watermark_img = watermark_img > 128 # Convert to binary image (0 or 255 -> True or False)

    s_box = generate_s_box(p, x0)

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

            encrypted_watermark_block = encrypt_watermark(resized_watermark_block.astype(int), s_box)

            for row in range(block_size):
                for col in range(block_size):
                    host_pixel = bin(embedded_img[i + row, j + col])[2:].zfill(8)
                    encrypted_watermark_value = int(encrypted_watermark_block[row, col])

                    if edge_magnitude[row, col] >= max_edge / 2:
                        # Alter 1 LSB
                        watermark_bit = encrypted_watermark_value % 2
                        embedded_pixel = host_pixel[:-1] + str(watermark_bit)
                    else:
                        # Alter 2 LSBs
                        watermark_bits = bin(encrypted_watermark_value)[-2:].zfill(2)
                        embedded_pixel = host_pixel[:-2] + watermark_bits

                    embedded_img[i + row, j + col] = int(embedded_pixel, 2)

    cv2.imwrite(embedded_image_path, embedded_img)
    print(f"Watermark embedded successfully in {embedded_image_path}")

def extract_watermark(embedded_image_path, original_image_path, extracted_watermark_path, p, x0):
    """Extracts the watermark from the embedded image."""
    embedded_img = cv2.imread(embedded_image_path, cv2.IMREAD_GRAYSCALE)
    original_img = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)

    s_box = generate_s_box(p, x0)

    host_h, host_w = embedded_img.shape
    block_size = 16
    extracted_watermark = np.zeros((host_h // block_size * 32 // block_size, host_w // block_size * 32 // block_size), dtype=np.uint8) # Assuming original watermark was 32x32

    for i in range(0, host_h - block_size + 1, block_size):
        for j in range(0, host_w - block_size + 1, block_size):
            embedded_block = embedded_img[i:i + block_size, j:j + block_size]
            original_block = original_img[i:i + block_size, j:j + block_size]

            edge_magnitude = compass_edge_detector(original_block)
            max_edge = np.max(edge_magnitude)

            extracted_watermark_block_encrypted = np.zeros((block_size, block_size), dtype=int)

            for row in range(block_size):
                for col in range(block_size):
                    embedded_pixel = bin(embedded_block[row, col])[2:].zfill(8)

                    if edge_magnitude[row, col] >= max_edge / 2:
                        # Extract 1 LSB
                        watermark_bit_encrypted = int(embedded_pixel[-1])
                    else:
                        # Extract 2 LSBs
                        watermark_bit_encrypted = int(embedded_pixel[-2:], 2)

                    extracted_watermark_block_encrypted[row, col] = watermark_bit_encrypted

            decrypted_watermark_block = decrypt_watermark(extracted_watermark_block_encrypted, s_box)

            # Place the extracted block into the larger watermark image
            wm_i_start = (i // block_size) * (32 // block_size)
            wm_i_end = wm_i_start + 1
            wm_j_start = (j // block_size) * (32 // block_size)
            wm_j_end = wm_j_start + 1

            extracted_watermark[wm_i_start, wm_j_start] = decrypted_watermark_block[0,0] * 255 # Assuming watermark block is binary

    cv2.imwrite(extracted_watermark_path, extracted_watermark)
    print(f"Watermark extracted successfully to {extracted_watermark_path}")

def verify_watermark(original_watermark_path, extracted_watermark_path):
    """Verifies the extracted watermark against the original."""
    original_wm = cv2.imread(original_watermark_path, cv2.IMREAD_GRAYSCALE)
    extracted_wm = cv2.imread(extracted_watermark_path, cv2.IMREAD_GRAYSCALE)

    if original_wm is None or extracted_wm is None:
        print("Error: Could not load watermark images for verification.")
        return

    # Resize extracted watermark to match original size
    extracted_wm_resized = cv2.resize(extracted_wm, original_wm.shape[::-1], interpolation=cv2.INTER_NEAREST)

    if original_wm.shape != extracted_wm_resized.shape:
        print("Error: Watermark shapes do not match for verification.")
        return

    # Threshold to make sure both are binary
    _, original_wm_binary = cv2.threshold(original_wm, 128, 255, cv2.THRESH_BINARY)
    _, extracted_wm_binary = cv2.threshold(extracted_wm_resized, 128, 255, cv2.THRESH_BINARY)

    difference = cv2.absdiff(original_wm_binary, extracted_wm_binary)
    non_zero_pixels = np.count_nonzero(difference)
    total_pixels = original_wm.size
    similarity = (total_pixels - non_zero_pixels) / total_pixels * 100

    print(f"Watermark Verification:")
    print(f"  Similarity: {similarity:.2f}%")

if __name__ == "__main__":
    host_image_path = "lena.png"  # Replace with your host image path
    watermark_image_path = "watermark.png"  # Replace with your watermark image path (should be small and binary)
    embedded_image_path = "embedded_image.png"
    extracted_watermark_path = "extracted_watermark.png"

    # PWLCM parameters
    p = 0.4
    x0 = 0.7

    # Embed the watermark
    embed_watermark(host_image_path, watermark_image_path, embedded_image_path, p, x0)

    # Extract the watermark
    extract_watermark(embedded_image_path, host_image_path, extracted_watermark_path, p, x0)

    # Verify the extracted watermark
    verify_watermark(watermark_image_path, extracted_watermark_path)