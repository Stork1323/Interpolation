import numpy as np
from PIL import Image

def nearest_neighbor_interpolation(image, scale_x, scale_y):
    # Get the size of the original image
    width, height = image.size
    # Calculate the new size
    new_width = int(width * scale_x)
    new_height = int(height * scale_y)

    # Create a new blank image
    rescaled_image = Image.new("RGB", (new_width, new_height))

    # Get the pixel data of the original image
    original_pixels = image.load()

    # For each pixel in the new image, find the corresponding nearest pixel in the original image
    for y in range(new_height):
        for x in range(new_width):
            # Find the nearest pixel in the original image
            orig_x = int(x / scale_x)
            orig_y = int(y / scale_y)

            # Get the pixel value from the original image
            pixel_value = original_pixels[orig_x, orig_y]

            # Set the pixel value in the new image
            rescaled_image.putpixel((x, y), pixel_value)

    return rescaled_image
def bilinear_interpolation(image, scale_x, scale_y):
    # Get the size of the original image
    width, height = image.size
    # Calculate the new size
    new_width = int(width * scale_x)
    new_height = int(height * scale_y)

    # Create a new blank image
    rescaled_image = Image.new("RGB", (new_width, new_height))

    # Get the pixel data of the original image
    original_pixels = image.load()

    for y in range(new_height):
        for x in range(new_width):
            # Find the coordinates of the four nearest pixels in the original image
            orig_x = x / scale_x
            orig_y = y / scale_y

            x0 = int(orig_x)
            x1 = min(x0 + 1, width - 1)
            y0 = int(orig_y)
            y1 = min(y0 + 1, height - 1)

            # Calculate the distances between the original coordinates and the four pixels
            dx = orig_x - x0
            dy = orig_y - y0

            # Get the pixel values of the four neighbors
            top_left = np.array(original_pixels[x0, y0])
            top_right = np.array(original_pixels[x1, y0])
            bottom_left = np.array(original_pixels[x0, y1])
            bottom_right = np.array(original_pixels[x1, y1])

            # Bilinear interpolation formula
            top = top_left * (1 - dx) + top_right * dx
            bottom = bottom_left * (1 - dx) + bottom_right * dx
            pixel_value = top * (1 - dy) + bottom * dy

            # Set the interpolated pixel value in the new image
            rescaled_image.putpixel((x, y), tuple(pixel_value.astype(int)))

    return rescaled_image

def bicubic_interpolation(image, scale_x, scale_y):
    # Get the size of the original image
    width, height = image.size
    # Calculate the new size
    new_width = int(width * scale_x)
    new_height = int(height * scale_y)

    # Create a new blank image
    rescaled_image = Image.new("RGB", (new_width, new_height))

    # Get the pixel data of the original image
    original_pixels = image.load()

    # Bicubic kernel function
    def cubic(x):
        x = abs(x)
        if x <= 1:
            return 1 - 2*x*x + x*x*x
        elif x < 2:
            return 4 - 8*x + 5*x*x - x*x*x
        return 0

    # Apply bicubic interpolation
    for y in range(new_height):
        for x in range(new_width):
            # Find the position in the original image
            orig_x = x / scale_x
            orig_y = y / scale_y

            # Get the integer part of the original position
            x_int = int(orig_x)
            y_int = int(orig_y)

            # Initialize RGB values
            R, G, B = 0, 0, 0

            # Bicubic interpolation involves a 4x4 grid of neighboring pixels
            for m in range(-1, 3):
                for n in range(-1, 3):
                    # Find the pixel value at (x_int + m, y_int + n)
                    px = min(max(x_int + m, 0), width - 1)
                    py = min(max(y_int + n, 0), height - 1)

                    # Get the pixel value
                    pixel = original_pixels[px, py]

                    # Calculate the weight based on the distance
                    weight = cubic(orig_x - px) * cubic(orig_y - py)

                    # Accumulate the weighted pixel values
                    R += pixel[0] * weight
                    G += pixel[1] * weight
                    B += pixel[2] * weight

            # Assign the interpolated pixel value to the new image
            rescaled_image.putpixel((x, y), (int(R), int(G), int(B)))

    return rescaled_image


# Load an image
image = Image.open("../input_image/input.jpg")  # Replace with your image path

# Define scaling factors (e.g., scale by 2x in both directions)
scale_x = 2.0
scale_y = 2.0

# Rescale the image using nearest neighbor interpolation
rescaled_image = nearest_neighbor_interpolation(image, scale_x, scale_y)

# Save the rescaled image
rescaled_image.save("../output_image/nearest_neighbor.jpg")

rescaled_image = bilinear_interpolation(image, scale_x, scale_y)
rescaled_image.save("../output_image/bilinear.jpg")

rescaled_image = bicubic_interpolation(image, scale_x, scale_y)
rescaled_image.save("../output_image/bicubic.jpg")
