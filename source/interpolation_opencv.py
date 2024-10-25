import cv2
from matplotlib import pyplot as plt

# Load an image using OpenCV
image = cv2.imread('../input_image/input.jpg')  # Replace with your image path

# Define the scaling factors
scale_x = 2.0  # Scaling along x-axis (e.g., 2x)
scale_y = 2.0  # Scaling along y-axis (e.g., 2x)

# Resize using Nearest Neighbor Interpolation
resized_nearest = cv2.resize(image, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_NEAREST)

# Resize using Bilinear Interpolation
resized_bilinear = cv2.resize(image, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)

# Resize using Bicubic Interpolation
resized_bicubic = cv2.resize(image, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_CUBIC)

# Convert the images from BGR to RGB for proper display in matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
resized_nearest_rgb = cv2.cvtColor(resized_nearest, cv2.COLOR_BGR2RGB)
resized_bilinear_rgb = cv2.cvtColor(resized_bilinear, cv2.COLOR_BGR2RGB)
resized_bicubic_rgb = cv2.cvtColor(resized_bicubic, cv2.COLOR_BGR2RGB)

# Plot and compare the results using matplotlib

# Original image

# Nearest Neighbor result
cv2.imwrite('../output_image/nearest_neighbor_opencv.jpg',resized_nearest_rgb)

# Bilinear result
cv2.imwrite('../output_image/bilinear_opencv.jpg',resized_bilinear_rgb)
# Bicubic result
cv2.imwrite('../output_image/bicubic_opencv.jpg',resized_bicubic_rgb)

