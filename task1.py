import cv2
import numpy as np
import matplotlib.pyplot as plt

#ssd

# Load the left and right images
left_img_path = "images/left.png"
right_img_path = "images/right.png"

left_img = cv2.imread(left_img_path, cv2.IMREAD_GRAYSCALE)
right_img = cv2.imread(right_img_path, cv2.IMREAD_GRAYSCALE)

# Define block size and disparity range
block_size = 5  
max_disparity = 30  

# Get image dimensions
height, width = left_img.shape

# Initialize disparity map with zeros
disparity_map = np.zeros((height, width), dtype=np.uint8)

# Perform block matching using SAD manually
for y in range(block_size//2, height - block_size//2):
    for x in range(block_size//2, width - block_size//2):

        best_offset = 0
        min_sad = float('inf')

        # Extract reference block from left image
        left_block = left_img[y - block_size//2 : y + block_size//2 + 1,
                              x - block_size//2 : x + block_size//2 + 1]

        # Search in the right image within disparity range
        for d in range(max_disparity):
            if x - d - block_size//2 < 0:
                break  # Avoid searching out of image bounds

            right_block = right_img[y - block_size//2 : y + block_size//2 + 1,
                                    x - d - block_size//2 : x - d + block_size//2 + 1]

            # Compute SAD
            sad = np.sum(np.abs(left_block.astype(np.int16) - right_block.astype(np.int16)))

            # Store the best match
            if sad < min_sad:
                min_sad = sad
                best_offset = d

        # Assign disparity value (scaled for better visualization)
        disparity_map[y, x] = best_offset * (255 // max_disparity)

# Apply a color map (Red for close, Blue for far)
disparity_color = cv2.applyColorMap(disparity_map, cv2.COLORMAP_JET)

# Display the colorized disparity map
plt.figure(figsize=(10, 5))
plt.imshow(cv2.cvtColor(disparity_color, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for correct colors
plt.axis("off")
plt.show()
