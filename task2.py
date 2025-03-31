import cv2
import numpy as np

# Load the π image
image_path = "pi_image.png"
img = cv2.imread(image_path, 0)

# Convert image to NumPy array 
img_array = np.array(img)

# Multiply each pixel by 10*pi and take floor
transformed_array = np.floor(img_array * (10 * np.pi))

# Flatten and sort in descending order
sorted_values = np.sort(transformed_array.flatten())[::-1]

# Extract the top 4 values for the 2x2 filter
filter_kernel = sorted_values[:4].reshape(2, 2)

# Convert to uint8 for bitwise operations
filter_kernel = filter_kernel.astype(np.uint8)

# Print the extracted filter
print("2x2 Filter:")
print(filter_kernel)

# Load Picasso's distorted image
picasso_img = cv2.imread('artwork_picasso.png', 0)

# Convert Picasso image to NumPy array 
img_picasso = np.array(picasso_img)

# Function to apply bitwise operations (OR, AND, XOR)
def apply_filter(img, operation):
    filtered_img = img.copy()  # Use a copy to avoid modifying the original image
    rows, cols = img.shape
    f_rows, f_cols = filter_kernel.shape

    # Slide the filter over the image with a step of filter width (2)
    for i in range(0, rows - f_rows + 1, f_rows):
        for j in range(0, cols - f_cols + 1, f_cols):
            if operation == "OR":
                filtered_img[i:i+f_rows, j:j+f_cols] |= filter_kernel
            elif operation == "AND":
                filtered_img[i:i+f_rows, j:j+f_cols] &= filter_kernel
            elif operation == "XOR":
                filtered_img[i:i+f_rows, j:j+f_cols] ^= filter_kernel

    return filtered_img

# Apply all three operations
or_img = apply_filter(img_picasso, "OR")
and_img = apply_filter(img_picasso, "AND")
xor_img = apply_filter(img_picasso, "XOR")

# Resize the images to 100x100 without saving
resized_or = cv2.resize(or_img, (100, 100))
resized_and = cv2.resize(and_img, (100, 100))
resized_xor = cv2.resize(xor_img, (100, 100))

# Display images
cv2.imshow("Resized OR", resized_or)
cv2.imshow("Resized AND", resized_and)
cv2.imshow("Resized XOR", resized_xor)

# Load collage image in grayscale
collage = cv2.imread("collage.png", 0)

#template matching
def manual_template_matching(collage, template):
    collage_h, collage_w = collage.shape
    template_h, template_w = template.shape
    
    best_match = (0, 0)
    min_ssd = float('inf')
    
    for y in range(collage_h - template_h + 1):
        for x in range(collage_w - template_w + 1):
            region = collage[y:y+template_h, x:x+template_w]
            ssd = np.sum((region - template) ** 2)
            
            if ssd < min_ssd:
                min_ssd = ssd
                best_match = (x, y)
    
    return best_match
# Load and resize templates
template1 = cv2.resize(or_img, (100, 100))
template2 = cv2.resize(and_img, (100, 100))
template3 = cv2.resize(xor_img, (100, 100))

# Find best match locations
match1 = manual_template_matching(collage, template1)
match2 = manual_template_matching(collage, template2)
match3 = manual_template_matching(collage, template3)

# Print coordinates
print("Template 1 Best Match:", match1)
print("Template 2 Best Match:", match2)
print("Template 3 Best Match:", match3)

# Compute the final password correctly
best_match_x, best_match_y = match1  # We use match1's coordinates as the best match
password = np.floor((best_match_x + best_match_y) * np.pi)
print("Final Password:", int(password))

best_match_x, best_match_y = match2  # We use match1's coordinates as the best match
password = np.floor((best_match_x + best_match_y) * np.pi)
print("Final Password2:", int(password))

best_match_x, best_match_y = match3  # We use match1's coordinates as the best match
password = np.floor((best_match_x + best_match_y) * np.pi)
print("Final Password3:", int(password))

# Wait for key press & close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()
