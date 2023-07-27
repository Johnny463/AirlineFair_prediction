import numpy as np
from PIL import Image
import os





def crop_image(image_array, new_width, new_height):
    # Get the original dimensions of the image
    original_height, original_width = image_array.shape[:2]

    # Calculate the aspect ratios
    original_aspect_ratio = original_width / original_height
    new_aspect_ratio = new_width / new_height

    # Determine the region to crop
    if original_aspect_ratio > new_aspect_ratio:
        # Crop the width to maintain the new aspect ratio
        crop_width = int(new_height * original_aspect_ratio)
        crop_left = (original_width - crop_width) // 2
        crop_right = crop_left + crop_width
        crop_box = (crop_left, 0, crop_right, original_height)
    else:
        # Crop the height to maintain the new aspect ratio
        crop_height = int(new_width / original_aspect_ratio)
        crop_top = (original_height - crop_height) // 2
        crop_bottom = crop_top + crop_height
        crop_box = (0, crop_top, original_width, crop_bottom)

    # Crop the image
    cropped_image = Image.fromarray(image_array).crop(crop_box)

    return cropped_image

# Load the image using PIL (Python Imaging Library)
image = Image.open('image.jpg')

# Convert the PIL image to a NumPy array
image_array = np.array(image)

# Convert the image to grayscale
gray_image = np.dot(image_array[..., :3], [0.2989, 0.5870, 0.1140])

# Convert the values to integers between 0 and 255 (8-bit grayscale)
gray_image = np.round(gray_image).astype(np.uint8)


# Perform the stretch operation
new_width, new_height = 400, 133  # Replace with desired dimensions
stretched_pil_image = image.resize((new_width, new_height), Image.LANCZOS)

# Save the stretched image
filename = image.filename
base_directory = os.getcwd()
image_directory = os.path.join(base_directory, "images")
os.makedirs(image_directory, exist_ok=True)
stretched_file_path = os.path.join(image_directory, "stretched.jpg")  # Use the desired file extension, e.g., '.jpg'
stretched_pil_image.save(stretched_file_path)

# Perform the crop operation
new_width, new_height = 300, 60  # Replace with desired dimensions
cropped_image = crop_image(gray_image, new_width, new_height)

# Save the cropped image
cropped_file_path = os.path.join(image_directory, "cropped_"+filename)
cropped_image.save(cropped_file_path)
