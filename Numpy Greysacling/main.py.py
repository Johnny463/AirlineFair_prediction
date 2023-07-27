import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load the image using PIL (Python Imaging Library)
image = Image.open('image.jpg')

# Convert the PIL image to a NumPy array
image_array = np.array(image)

# Convert the image to grayscale
gray_image = np.dot(image_array[..., :3], [0.2989, 0.5870, 0.1140])

# Convert the values to integers between 0 and 255 (8-bit grayscale)
gray_image = np.round(gray_image).astype(np.uint8)

# If you want to extract a specific region (e.g., a 100x100 pixel region starting from (100, 1000)):# # Extract the region

# # Display the grayscale image
# plt.imshow(gray_image, cmap='gray')
# plt.title('Grayscale Image')
# plt.axis('off')  # Turn off axis ticks and labels
# plt.show()


#Converted arrayed image into PIL image
pil_image = Image.fromarray(gray_image)

#Showing image in image form
pil_image.show()